use std::io;
use std::sync::Mutex;

use common::OwnedBytes;
use zstd::bulk::{compress_to_buffer, decompress_to_buffer};
use zstd::zstd_safe::{self, CCtx, CDict, DCtx, DDict};
use zstd::DEFAULT_COMPRESSION_LEVEL;

const COUNT_SIZE: usize = std::mem::size_of::<u32>();

/// A compressed block is the uncompressed length (u32 LE) followed by a single zstd frame.
#[inline]
fn compress_with(
    uncompressed: &[u8],
    compressed: &mut Vec<u8>,
    compress_frame: impl FnOnce(&[u8], &mut [u8]) -> io::Result<usize>,
) -> io::Result<()> {
    let max_size = zstd_safe::compress_bound(uncompressed.len()) + COUNT_SIZE;

    compressed.clear();
    compressed.resize(max_size, 0);

    let compressed_size = compress_frame(uncompressed, &mut compressed[COUNT_SIZE..])?;

    compressed[0..COUNT_SIZE].copy_from_slice(&(uncompressed.len() as u32).to_le_bytes());
    compressed.resize(compressed_size + COUNT_SIZE, 0);

    Ok(())
}

#[inline]
fn decompress_with(
    compressed: &[u8],
    decompressed: &mut Vec<u8>,
    decompress_frame: impl FnOnce(&[u8], &mut [u8]) -> io::Result<usize>,
) -> io::Result<()> {
    let uncompressed_size = u32::from_le_bytes(
        compressed
            .get(..COUNT_SIZE)
            .ok_or(io::ErrorKind::InvalidData)?
            .try_into()
            .unwrap(),
    ) as usize;

    decompressed.clear();
    decompressed.resize(uncompressed_size, 0);

    let decompressed_size = decompress_frame(&compressed[COUNT_SIZE..], decompressed)?;

    if decompressed_size != uncompressed_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "doc store block not completely decompressed, data corruption".to_string(),
        ));
    }

    Ok(())
}

fn zstd_error(kind: io::ErrorKind, code: zstd_safe::ErrorCode) -> io::Error {
    io::Error::new(kind, zstd_safe::get_error_name(code).to_string())
}

#[inline]
pub fn compress(
    uncompressed: &[u8],
    compressed: &mut Vec<u8>,
    compression_level: Option<i32>,
) -> io::Result<()> {
    compress_with(uncompressed, compressed, |src, dst| {
        compress_to_buffer(
            src,
            dst,
            compression_level.unwrap_or(DEFAULT_COMPRESSION_LEVEL),
        )
    })
}

#[inline]
pub fn decompress(compressed: &[u8], decompressed: &mut Vec<u8>) -> io::Result<()> {
    decompress_with(compressed, decompressed, decompress_to_buffer)
}

/// Trains a zstd dictionary of at most `max_dict_size` bytes.
///
/// `samples` is the concatenation of the training samples, whose lengths are `sample_sizes`.
pub fn train_dictionary(
    samples: &[u8],
    sample_sizes: &[usize],
    max_dict_size: usize,
) -> io::Result<Vec<u8>> {
    zstd::dict::from_continuous(samples, sample_sizes, max_dict_size)
}

/// Compresses doc store blocks with a dictionary, reusing the same compression context.
pub struct ZstdDictEncoder {
    dict: Vec<u8>,
    cdict: CDict<'static>,
    cctx: CCtx<'static>,
}

impl ZstdDictEncoder {
    pub fn new(dict: Vec<u8>, compression_level: Option<i32>) -> io::Result<Self> {
        let cdict = CDict::try_create(
            &dict,
            compression_level.unwrap_or(DEFAULT_COMPRESSION_LEVEL),
        )
        .ok_or_else(|| io::Error::other("failed to load zstd dictionary for compression"))?;
        let cctx = CCtx::try_create()
            .ok_or_else(|| io::Error::other("failed to create zstd compression context"))?;
        Ok(ZstdDictEncoder { dict, cdict, cctx })
    }

    pub fn dict(&self) -> &[u8] {
        &self.dict
    }

    pub fn into_dict(self) -> Vec<u8> {
        self.dict
    }

    pub fn compress(&mut self, uncompressed: &[u8], compressed: &mut Vec<u8>) -> io::Result<()> {
        let ZstdDictEncoder { cdict, cctx, .. } = self;
        compress_with(uncompressed, compressed, |src, dst| {
            cctx.compress_using_cdict(dst, src, cdict)
                .map_err(|code| zstd_error(io::ErrorKind::Other, code))
        })
    }
}

/// Maximum number of idle decompression contexts kept by a `ZstdDictDecoder`.
const MAX_POOLED_DCTX: usize = 4;

/// Prepared (digested) dictionary used to decompress the blocks of one doc store.
///
/// It is parsed once when the store is opened and shared by all reads. Decompression contexts
/// are reused: zstd switches to a slower, prefetching decoder whenever a context sees a
/// dictionary for the first time.
pub struct ZstdDictDecoder {
    ddict: DDict<'static>,
    dctx_pool: Mutex<Vec<DCtx<'static>>>,
}

impl ZstdDictDecoder {
    /// `checksum` is the CRC32 of `dict`.
    pub fn open(dict: OwnedBytes, checksum: u32) -> io::Result<Self> {
        // Frames record the id of the dictionary they were compressed with, so requiring a
        // proper zstd dictionary (with an id) turns a wrong dictionary into a decoding error.
        if zstd_safe::get_dict_id_from_dict(&dict).is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "doc store zstd dictionary is corrupted (missing magic number or dictionary id)",
            ));
        }
        if crc32fast::hash(&dict) != checksum {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "doc store zstd dictionary is corrupted (checksum mismatch)",
            ));
        }
        let ddict = DDict::try_create(&dict).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "doc store zstd dictionary is corrupted",
            )
        })?;
        Ok(ZstdDictDecoder {
            ddict,
            dctx_pool: Mutex::new(Vec::new()),
        })
    }

    pub fn decompress(&self, compressed: &[u8], decompressed: &mut Vec<u8>) -> io::Result<()> {
        let pooled_dctx = self.dctx_pool.lock().unwrap().pop();
        let mut dctx = match pooled_dctx {
            Some(dctx) => dctx,
            None => DCtx::try_create()
                .ok_or_else(|| io::Error::other("failed to create zstd decompression context"))?,
        };
        let result = decompress_with(compressed, decompressed, |src, dst| {
            dctx.decompress_using_ddict(dst, src, &self.ddict)
                .map_err(|code| zstd_error(io::ErrorKind::InvalidData, code))
        });
        let mut dctx_pool = self.dctx_pool.lock().unwrap();
        if dctx_pool.len() < MAX_POOLED_DCTX {
            dctx_pool.push(dctx);
        }
        result
    }
}
