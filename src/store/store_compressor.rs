use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::{io, thread};

use common::{BinarySerializable, CountingWriter, TerminatingWrite};

#[cfg(feature = "zstd-compression")]
use super::compression_zstd_block::{train_dictionary, ZstdDictEncoder};
use super::DOC_STORE_VERSION;
use crate::directory::WritePtr;
use crate::store::footer::DocStoreFooter;
use crate::store::index::{Checkpoint, SkipIndexBuilder};
use crate::store::{Compressor, Decompressor, StoreReader};
use crate::DocId;

pub struct BlockCompressor {
    variants: BlockCompressorVariants,
    /// Memory held by the compressor, updated by the thread doing the compression.
    mem_usage: Arc<AtomicUsize>,
}

// The struct wrapping an enum is just here to keep the
// impls private.
enum BlockCompressorVariants {
    SameThread(BlockCompressorImpl),
    DedicatedThread(DedicatedThreadBlockCompressorImpl),
}

impl BlockCompressor {
    pub fn new(compressor: Compressor, wrt: WritePtr, dedicated_thread: bool) -> io::Result<Self> {
        let block_compressor_impl = BlockCompressorImpl::new(compressor, wrt)?;
        let mem_usage = block_compressor_impl.mem_usage.clone();
        let variants = if dedicated_thread {
            BlockCompressorVariants::DedicatedThread(DedicatedThreadBlockCompressorImpl::new(
                block_compressor_impl,
            )?)
        } else {
            BlockCompressorVariants::SameThread(block_compressor_impl)
        };
        Ok(BlockCompressor {
            variants,
            mem_usage,
        })
    }

    /// Memory held by the compressor itself, e.g. the blocks buffered to train a zstd
    /// dictionary. Blocks in flight to a dedicated thread are not accounted for.
    pub fn mem_usage(&self) -> usize {
        self.mem_usage.load(Ordering::Relaxed)
    }

    pub fn compress_block_and_write(
        &mut self,
        bytes: &[u8],
        num_docs_in_block: u32,
    ) -> io::Result<()> {
        match &mut self.variants {
            BlockCompressorVariants::SameThread(block_compressor) => {
                block_compressor.compress_block_and_write(bytes, num_docs_in_block)?;
            }
            BlockCompressorVariants::DedicatedThread(different_thread_block_compressor) => {
                different_thread_block_compressor
                    .compress_block_and_write(bytes, num_docs_in_block)?;
            }
        }
        Ok(())
    }

    pub fn stack_reader(&mut self, store_reader: StoreReader) -> io::Result<()> {
        match &mut self.variants {
            BlockCompressorVariants::SameThread(block_compressor) => {
                block_compressor.stack(store_reader)?;
            }
            BlockCompressorVariants::DedicatedThread(different_thread_block_compressor) => {
                different_thread_block_compressor.stack_reader(store_reader)?;
            }
        }
        Ok(())
    }

    pub fn close(self) -> io::Result<()> {
        match self.variants {
            BlockCompressorVariants::SameThread(block_compressor) => block_compressor.close(),
            BlockCompressorVariants::DedicatedThread(different_thread_block_compressor) => {
                different_thread_block_compressor.close()
            }
        }
    }
}

struct BlockCompressorImpl {
    compressor: Compressor,
    first_doc_in_block: DocId,
    offset_index_writer: SkipIndexBuilder,
    intermediary_buffer: Vec<u8>,
    writer: CountingWriter<WritePtr>,
    /// `Some` iff the compressor trains a zstd dictionary.
    #[cfg(feature = "zstd-compression")]
    zstd_dict: Option<ZstdDictState>,
    mem_usage: Arc<AtomicUsize>,
}

impl BlockCompressorImpl {
    fn new(compressor: Compressor, writer: WritePtr) -> io::Result<Self> {
        Ok(Self {
            compressor,
            first_doc_in_block: 0,
            offset_index_writer: SkipIndexBuilder::new(),
            intermediary_buffer: Vec::new(),
            writer: CountingWriter::wrap(writer),
            #[cfg(feature = "zstd-compression")]
            zstd_dict: ZstdDictState::for_compressor(compressor)?,
            mem_usage: Arc::default(),
        })
    }

    #[cfg(feature = "zstd-compression")]
    fn update_mem_usage(&self) {
        let mem_usage = match &self.zstd_dict {
            Some(ZstdDictState::Sampling(sample)) => sample.mem_usage(),
            Some(ZstdDictState::Ready(Some(encoder))) => encoder.mem_usage(),
            Some(ZstdDictState::Ready(None)) | None => 0,
        };
        self.mem_usage.store(mem_usage, Ordering::Relaxed);
    }

    fn compress_block_and_write(&mut self, data: &[u8], num_docs_in_block: u32) -> io::Result<()> {
        assert!(num_docs_in_block > 0);
        #[cfg(feature = "zstd-compression")]
        match &mut self.zstd_dict {
            Some(ZstdDictState::Sampling(sample)) => {
                sample.push(data, num_docs_in_block);
                if sample.is_complete() {
                    self.finish_dict_sampling()?;
                } else {
                    self.update_mem_usage();
                }
                return Ok(());
            }
            Some(ZstdDictState::Ready(Some(encoder))) => {
                encoder.compress(data, &mut self.intermediary_buffer)?;
                return self.write_intermediary_buffer(num_docs_in_block);
            }
            Some(ZstdDictState::Ready(None)) | None => {}
        }
        self.intermediary_buffer.clear();
        self.compressor
            .compress_into(data, &mut self.intermediary_buffer)?;
        self.write_intermediary_buffer(num_docs_in_block)
    }

    fn write_intermediary_buffer(&mut self, num_docs_in_block: u32) -> io::Result<()> {
        let start_offset = self.writer.written_bytes() as usize;
        self.writer.write_all(&self.intermediary_buffer)?;
        let end_offset = self.writer.written_bytes() as usize;

        self.register_checkpoint(Checkpoint {
            doc_range: self.first_doc_in_block..self.first_doc_in_block + num_docs_in_block,
            byte_range: start_offset..end_offset,
        });
        Ok(())
    }

    /// Trains the dictionary on the buffered blocks, decides whether to use it, and writes the
    /// buffered blocks.
    #[cfg(feature = "zstd-compression")]
    fn finish_dict_sampling(&mut self) -> io::Result<()> {
        let sample = match self.zstd_dict.take() {
            Some(ZstdDictState::Sampling(sample)) => sample,
            other => {
                self.zstd_dict = other;
                return Ok(());
            }
        };
        let Compressor::Zstd(zstd_compressor) = self.compressor else {
            unreachable!("only zstd compressors train a dictionary");
        };
        let encoder = match sample.train_and_compress(zstd_compressor.compression_level)? {
            Some((encoder, compressed_blocks)) => {
                for (compressed_block, num_docs_in_block) in compressed_blocks {
                    self.intermediary_buffer = compressed_block;
                    self.write_intermediary_buffer(num_docs_in_block)?;
                }
                Some(encoder)
            }
            None => {
                for (block, num_docs_in_block) in sample.blocks() {
                    self.compressor
                        .compress_into(block, &mut self.intermediary_buffer)?;
                    self.write_intermediary_buffer(num_docs_in_block)?;
                }
                None
            }
        };
        self.zstd_dict = Some(ZstdDictState::Ready(encoder));
        self.update_mem_usage();
        Ok(())
    }

    fn register_checkpoint(&mut self, checkpoint: Checkpoint) {
        self.offset_index_writer.insert(checkpoint.clone());
        self.first_doc_in_block = checkpoint.doc_range.end;
    }

    /// Stacks a store reader on top of the documents written so far.
    /// This method is an optimization compared to iterating over the documents
    /// in the store and adding them one by one, as the store's data will
    /// not be decompressed and then recompressed.
    fn stack(&mut self, store_reader: StoreReader) -> io::Result<()> {
        if !self.compressor.can_stack(store_reader.decompressor()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "cannot stack a doc store compressed with {:?} into a doc store compressed \
                     with {:?}",
                    store_reader.decompressor(),
                    self.compressor
                ),
            ));
        }
        let doc_shift = self.first_doc_in_block;
        let start_shift = self.writer.written_bytes() as usize;

        // just bulk write all of the block of the given reader.
        self.writer
            .write_all(store_reader.block_data()?.as_slice())?;

        // concatenate the index of the `store_reader`, after translating
        // its start doc id and its start file offset.
        for mut checkpoint in store_reader.block_checkpoints() {
            checkpoint.doc_range.start += doc_shift;
            checkpoint.doc_range.end += doc_shift;
            checkpoint.byte_range.start += start_shift;
            checkpoint.byte_range.end += start_shift;
            self.register_checkpoint(checkpoint);
        }
        Ok(())
    }

    /// Writes the dictionary (if any), the skip index and the footer.
    ///
    /// Layout of a doc store file:
    /// - the compressed blocks,
    /// - the zstd dictionary, only if the footer's decompressor is `ZstdDict`,
    /// - the skip index, starting at the offset recorded in the footer,
    /// - the footer (see `DocStoreFooter`), which also records the dictionary length.
    fn close(mut self) -> io::Result<()> {
        #[cfg(feature = "zstd-compression")]
        let dict = {
            self.finish_dict_sampling()?;
            match self.zstd_dict.take() {
                Some(ZstdDictState::Ready(Some(encoder))) => Some(encoder.into_dict()),
                _ => None,
            }
        };
        #[cfg(feature = "zstd-compression")]
        if let Some(dict) = &dict {
            self.writer.write_all(dict)?;
        }
        let header_offset: u64 = self.writer.written_bytes();
        let docstore_footer = DocStoreFooter::new(
            header_offset,
            Decompressor::from(self.compressor),
            DOC_STORE_VERSION,
        );
        #[cfg(feature = "zstd-compression")]
        let docstore_footer = match &dict {
            Some(dict) => DocStoreFooter::with_dictionary(
                header_offset,
                u32::try_from(dict.len()).map_err(io::Error::other)?,
                crc32fast::hash(dict),
            ),
            None => docstore_footer,
        };
        self.offset_index_writer.serialize_into(&mut self.writer)?;
        docstore_footer.serialize(&mut self.writer)?;
        self.writer.terminate()
    }
}

/// The dictionary is trained once this many times its size has been buffered (zstd's
/// recommendation), with at most `MAX_SAMPLE_BYTES` buffered to bound memory.
#[cfg(feature = "zstd-compression")]
const SAMPLE_SIZE_RATIO: usize = 100;
#[cfg(feature = "zstd-compression")]
const MAX_SAMPLE_BYTES: usize = 16 << 20;
/// Stores smaller than this many times the dictionary size do not train a dictionary:
/// zstd's trainer produces poor dictionaries below that ratio.
#[cfg(feature = "zstd-compression")]
const MIN_SAMPLE_SIZE_RATIO: usize = 10;
/// Blocks are split into training samples of at most this size, and into at least
/// `MIN_NUM_TRAINING_SAMPLES` samples overall: zstd's trainer needs many samples.
#[cfg(feature = "zstd-compression")]
const MAX_TRAINING_SAMPLE_LEN: usize = 64 << 10;
#[cfg(feature = "zstd-compression")]
const MIN_NUM_TRAINING_SAMPLES: usize = 100;
/// A dictionary is kept only if it saves at least 1/MIN_DICT_GAIN_RATIO of the size.
#[cfg(feature = "zstd-compression")]
const MIN_DICT_GAIN_RATIO: usize = 50;
/// Memory used by zstd's dictionary trainer: with its default parameters, fastCover allocates
/// tables of 4 MiB, 4 MiB and 2 MiB, plus a compression context to evaluate dictionaries.
#[cfg(feature = "zstd-compression")]
const TRAINER_MEM_USAGE: usize = 12 << 20;

#[cfg(feature = "zstd-compression")]
enum ZstdDictState {
    /// The first blocks are kept uncompressed until there is enough data to train.
    Sampling(DictSample),
    /// `None` means that training failed or did not pay off: the store falls back to plain
    /// zstd blocks.
    Ready(Option<ZstdDictEncoder>),
}

#[cfg(feature = "zstd-compression")]
impl ZstdDictState {
    fn for_compressor(compressor: Compressor) -> io::Result<Option<ZstdDictState>> {
        let Compressor::Zstd(zstd_compressor) = compressor else {
            return Ok(None);
        };
        let Some(dict_size) = zstd_compressor.dict_size else {
            return Ok(None);
        };
        zstd_compressor
            .validate()
            .map_err(|msg| io::Error::new(io::ErrorKind::InvalidInput, msg))?;
        Ok(Some(ZstdDictState::Sampling(DictSample::new(dict_size))))
    }
}

#[cfg(feature = "zstd-compression")]
struct DictSample {
    dict_size: usize,
    target_len: usize,
    /// Concatenation of the buffered uncompressed blocks.
    data: Vec<u8>,
    /// Length and number of docs of each buffered block.
    blocks: Vec<(usize, u32)>,
}

#[cfg(feature = "zstd-compression")]
impl DictSample {
    fn new(dict_size: usize) -> Self {
        DictSample {
            dict_size,
            target_len: (dict_size * SAMPLE_SIZE_RATIO).min(MAX_SAMPLE_BYTES),
            data: Vec::new(),
            blocks: Vec::new(),
        }
    }

    fn push(&mut self, block: &[u8], num_docs_in_block: u32) {
        let needed = self.data.len() + block.len();
        if needed > self.data.capacity() {
            // Grow geometrically, but never beyond the sample target.
            let capacity = (self.data.capacity() * 2).clamp(needed, self.target_len.max(needed));
            self.data.reserve_exact(capacity - self.data.len());
        }
        self.data.extend_from_slice(block);
        self.blocks.push((block.len(), num_docs_in_block));
    }

    /// Memory held by the sample, plus the memory needed to train a dictionary on it once it
    /// is big enough: first the trainer's tables, then a compressed copy of the sample. The
    /// encoder itself is accounted for once it exists (see `ZstdDictEncoder::mem_usage`).
    fn mem_usage(&self) -> usize {
        let buffers = self.data.capacity() + self.blocks.capacity() * size_of::<(usize, u32)>();
        if self.data.len() < self.dict_size * MIN_SAMPLE_SIZE_RATIO {
            return buffers;
        }
        let compressed_copy = zstd::zstd_safe::compress_bound(self.data.len());
        buffers + TRAINER_MEM_USAGE.max(compressed_copy)
    }

    fn is_complete(&self) -> bool {
        self.data.len() >= self.target_len
    }

    fn blocks(&self) -> impl Iterator<Item = (&[u8], u32)> + '_ {
        let mut start = 0;
        self.blocks.iter().map(move |&(len, num_docs)| {
            let block = &self.data[start..start + len];
            start += len;
            (block, num_docs)
        })
    }

    fn train(&self, compression_level: Option<i32>) -> Option<ZstdDictEncoder> {
        if self.data.len() < self.dict_size * MIN_SAMPLE_SIZE_RATIO {
            return None;
        }
        let max_sample_len =
            (self.data.len() / MIN_NUM_TRAINING_SAMPLES).clamp(1, MAX_TRAINING_SAMPLE_LEN);
        let mut sample_sizes = Vec::new();
        for &(len, _) in &self.blocks {
            let num_pieces = len.div_ceil(max_sample_len).max(1);
            let piece_len = len / num_pieces;
            sample_sizes.extend(std::iter::repeat_n(piece_len, num_pieces - 1));
            sample_sizes.push(len - piece_len * (num_pieces - 1));
        }
        let dict = train_dictionary(&self.data, &sample_sizes, self.dict_size).ok()?;
        ZstdDictEncoder::new(dict, compression_level).ok()
    }

    /// Trains a dictionary and returns its encoder together with the buffered blocks compressed
    /// with it, or `None` if the store should use plain zstd blocks.
    ///
    /// The dictionary is kept only if it makes the buffered blocks significantly smaller. An
    /// incomplete sample holds the whole store, so the size of the dictionary itself is then
    /// accounted for too.
    #[allow(clippy::type_complexity)]
    fn train_and_compress(
        &self,
        compression_level: Option<i32>,
    ) -> io::Result<Option<(ZstdDictEncoder, Vec<(Vec<u8>, u32)>)>> {
        let Some(mut encoder) = self.train(compression_level) else {
            return Ok(None);
        };
        let mut compressed = Vec::new();
        let mut without_dict = 0;
        for (block, _) in self.blocks() {
            super::compression_zstd_block::compress(block, &mut compressed, compression_level)?;
            without_dict += compressed.len();
        }
        let dict_blocks = self
            .blocks()
            .map(|(block, num_docs)| {
                let mut compressed = Vec::new();
                encoder.compress(block, &mut compressed)?;
                Ok((compressed, num_docs))
            })
            .collect::<io::Result<Vec<_>>>()?;
        let dict_overhead = if self.is_complete() {
            0
        } else {
            encoder.dict().len()
        };
        let with_dict = dict_blocks
            .iter()
            .map(|(block, _)| block.len())
            .sum::<usize>()
            + dict_overhead;
        if with_dict + without_dict / MIN_DICT_GAIN_RATIO < without_dict {
            Ok(Some((encoder, dict_blocks)))
        } else {
            Ok(None)
        }
    }
}

// ---------------------------------
enum BlockCompressorMessage {
    CompressBlockAndWrite {
        block_data: Vec<u8>,
        num_docs_in_block: u32,
    },
    Stack(StoreReader),
}

struct DedicatedThreadBlockCompressorImpl {
    join_handle: Option<JoinHandle<io::Result<()>>>,
    tx: SyncSender<BlockCompressorMessage>,
}

impl DedicatedThreadBlockCompressorImpl {
    fn new(mut block_compressor: BlockCompressorImpl) -> io::Result<Self> {
        let (tx, rx): (
            SyncSender<BlockCompressorMessage>,
            Receiver<BlockCompressorMessage>,
        ) = sync_channel(3);
        let join_handle = thread::Builder::new()
            .name("docstore-compressor-thread".to_string())
            .spawn(move || {
                while let Ok(packet) = rx.recv() {
                    match packet {
                        BlockCompressorMessage::CompressBlockAndWrite {
                            block_data,
                            num_docs_in_block,
                        } => {
                            block_compressor
                                .compress_block_and_write(&block_data[..], num_docs_in_block)?;
                        }
                        BlockCompressorMessage::Stack(store_reader) => {
                            block_compressor.stack(store_reader)?;
                        }
                    }
                }
                block_compressor.close()?;
                Ok(())
            })?;
        Ok(DedicatedThreadBlockCompressorImpl {
            join_handle: Some(join_handle),
            tx,
        })
    }

    fn compress_block_and_write(&mut self, bytes: &[u8], num_docs_in_block: u32) -> io::Result<()> {
        self.send(BlockCompressorMessage::CompressBlockAndWrite {
            block_data: bytes.to_vec(),
            num_docs_in_block,
        })
    }

    fn stack_reader(&mut self, store_reader: StoreReader) -> io::Result<()> {
        self.send(BlockCompressorMessage::Stack(store_reader))
    }

    fn send(&mut self, msg: BlockCompressorMessage) -> io::Result<()> {
        if self.tx.send(msg).is_err() {
            harvest_thread_result(self.join_handle.take())?;
            return Err(io::Error::other("Unidentified error."));
        }
        Ok(())
    }

    fn close(self) -> io::Result<()> {
        drop(self.tx);
        harvest_thread_result(self.join_handle)
    }
}

/// Wait for the thread result to terminate and returns its result.
///
/// If the thread panicked, or if the result has already been harvested,
/// returns an explicit error.
fn harvest_thread_result(join_handle_opt: Option<JoinHandle<io::Result<()>>>) -> io::Result<()> {
    let join_handle = join_handle_opt.ok_or_else(|| io::Error::other("Thread already joined."))?;
    join_handle
        .join()
        .map_err(|_err| io::Error::other("Compressing thread panicked."))?
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::path::Path;

    use crate::directory::RamDirectory;
    use crate::store::store_compressor::BlockCompressor;
    use crate::store::Compressor;
    use crate::Directory;

    fn populate_block_compressor(mut block_compressor: BlockCompressor) -> io::Result<()> {
        block_compressor.compress_block_and_write(b"hello", 1)?;
        block_compressor.compress_block_and_write(b"happy", 1)?;
        block_compressor.close()?;
        Ok(())
    }

    #[test]
    fn test_block_store_compressor_impls_yield_the_same_result() {
        let ram_directory = RamDirectory::default();
        let path1 = Path::new("path1");
        let path2 = Path::new("path2");
        let wrt1 = ram_directory.open_write(path1).unwrap();
        let wrt2 = ram_directory.open_write(path2).unwrap();
        let block_compressor1 = BlockCompressor::new(Compressor::None, wrt1, true).unwrap();
        let block_compressor2 = BlockCompressor::new(Compressor::None, wrt2, false).unwrap();
        populate_block_compressor(block_compressor1).unwrap();
        populate_block_compressor(block_compressor2).unwrap();
        let data1 = ram_directory.open_read(path1).unwrap();
        let data2 = ram_directory.open_read(path2).unwrap();
        assert_eq!(data1.read_bytes().unwrap(), data2.read_bytes().unwrap());
    }
}
