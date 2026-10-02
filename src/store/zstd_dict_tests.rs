//! Tests of doc stores compressed with a trained zstd dictionary.

use std::path::Path;

use common::{BinarySerializable, FixedSize, OwnedBytes};

use super::footer::DocStoreFooter;
use super::tests::write_lorem_ipsum_store;
use super::{Compressor, Decompressor, DocStoreVersion, StoreReader, StoreWriter, ZstdCompressor};
use crate::directory::{Directory, FileSlice, RamDirectory};
use crate::indexer::NoMergePolicy;
use crate::schema::{self, Schema, TantivyDocument, Value, INDEXED, STORED};
use crate::{Index, IndexSettings, IndexWriter, Term};

const BLOCK_SIZE: usize = 16_384;
const DICT_SIZE: usize = 4_096;

fn zstd_dict(compression_level: Option<i32>, dict_size: usize) -> Compressor {
    Compressor::Zstd(ZstdCompressor {
        compression_level,
        dict_size: Some(dict_size),
    })
}

fn write_store(compressor: Compressor, num_docs: usize, separate_thread: bool) -> OwnedBytes {
    let directory = RamDirectory::create();
    let path = Path::new("store");
    write_lorem_ipsum_store(
        directory.open_write(path).unwrap(),
        num_docs,
        compressor,
        BLOCK_SIZE,
        separate_thread,
    );
    directory.open_read(path).unwrap().read_bytes().unwrap()
}

fn file(store_bytes: &OwnedBytes) -> FileSlice {
    FileSlice::from(store_bytes.to_vec())
}

fn footer(store_bytes: &OwnedBytes) -> DocStoreFooter {
    let (footer, _) = DocStoreFooter::extract_footer(file(store_bytes)).unwrap();
    footer
}

fn embedded_dict(store_bytes: &OwnedBytes) -> Vec<u8> {
    let footer = footer(store_bytes);
    let end = footer.offset as usize;
    store_bytes.as_slice()[end - footer.dict_len as usize..end].to_vec()
}

fn check_lorem_store(store_bytes: OwnedBytes, num_docs: usize) -> crate::Result<()> {
    let store = StoreReader::open(file(&store_bytes), 10)?;
    let schema = lorem_schema();
    let title = schema.get_field("title").unwrap();
    for i in 0..num_docs as u32 {
        let doc: TantivyDocument = store.get(i)?;
        assert_eq!(
            doc.get_first(title).unwrap().as_str(),
            Some(&*format!("Doc {i}"))
        );
    }
    for (i, doc) in store.iter::<TantivyDocument>(None).enumerate() {
        assert_eq!(
            doc?.get_first(title).unwrap().as_str(),
            Some(&*format!("Doc {i}"))
        );
    }
    Ok(())
}

fn lorem_schema() -> Schema {
    let mut schema_builder = Schema::builder();
    schema_builder.add_text_field("body", STORED);
    schema_builder.add_text_field("title", STORED);
    schema_builder.build()
}

#[test]
fn test_store_zstd_dict_roundtrip() -> crate::Result<()> {
    for separate_thread in [false, true] {
        for compression_level in [None, Some(9)] {
            // ~470KB of data: the dictionary is trained once 100 * DICT_SIZE bytes are
            // buffered, and the remaining blocks are compressed as they come.
            let store_bytes = write_store(
                zstd_dict(compression_level, DICT_SIZE),
                1_000,
                separate_thread,
            );
            let footer = footer(&store_bytes);
            assert_eq!(footer.decompressor, Decompressor::ZstdDict);
            assert!(footer.dict_len > 0 && footer.dict_len as usize <= DICT_SIZE);
            let store = StoreReader::open(file(&store_bytes), 10)?;
            assert_eq!(store.decompressor(), Decompressor::ZstdDict);
            // The dictionary is accounted for in the data usage.
            assert_eq!(
                store.space_usage().total().get_bytes() as usize,
                store_bytes.len() - DocStoreFooter::SIZE_IN_BYTES
            );
            check_lorem_store(store_bytes, 1_000)?;
        }
    }
    Ok(())
}

#[test]
fn test_store_zstd_dict_whole_store_in_sample() -> crate::Result<()> {
    for separate_thread in [false, true] {
        // ~95KB: more than 10 * DICT_SIZE but less than the sample target, so the dictionary
        // is trained when the store is closed. Small blocks make the dictionary worthwhile.
        let directory = RamDirectory::create();
        let path = Path::new("store");
        write_lorem_ipsum_store(
            directory.open_write(path)?,
            200,
            zstd_dict(None, DICT_SIZE),
            1_024,
            separate_thread,
        );
        let store_bytes = directory.open_read(path)?.read_bytes()?;
        assert_eq!(footer(&store_bytes).decompressor, Decompressor::ZstdDict);
        check_lorem_store(store_bytes, 200)?;
    }
    Ok(())
}

#[test]
fn test_store_zstd_dict_is_smaller_on_small_blocks() {
    let plain = write_store(Compressor::Zstd(ZstdCompressor::default()), 1_000, true);
    let with_dict = write_store(zstd_dict(None, DICT_SIZE), 1_000, true);
    assert!(with_dict.len() < plain.len());
}

#[test]
fn test_store_zstd_dict_small_store_falls_back_to_plain_zstd() -> crate::Result<()> {
    for num_docs in [0, 1, 10] {
        let store_bytes = write_store(zstd_dict(None, DICT_SIZE), num_docs, true);
        assert_eq!(footer(&store_bytes).decompressor, Decompressor::Zstd);
        assert_eq!(footer(&store_bytes).dict_len, 0);
        // The fallback writes exactly what a plain zstd store would, readable by older readers.
        let plain = write_store(Compressor::Zstd(ZstdCompressor::default()), num_docs, true);
        assert_eq!(store_bytes.as_slice(), plain.as_slice());
        check_lorem_store(store_bytes, num_docs)?;
    }
    Ok(())
}

#[test]
fn test_store_zstd_dict_falls_back_when_not_worthwhile() -> crate::Result<()> {
    let mut schema_builder = Schema::builder();
    let bytes_field = schema_builder.add_bytes_field("bytes", STORED);
    let schema = schema_builder.build();
    let directory = RamDirectory::create();
    let path = Path::new("store");
    let mut rng_state = 42u64;
    let mut docs = Vec::new();
    {
        let mut store_writer = StoreWriter::new(
            directory.open_write(path)?,
            zstd_dict(None, DICT_SIZE),
            BLOCK_SIZE,
            false,
        )?;
        // Incompressible data: a dictionary cannot help.
        for _ in 0..100 {
            let bytes: Vec<u8> = (0..1_000)
                .map(|_| {
                    rng_state = rng_state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (rng_state >> 33) as u8
                })
                .collect();
            let mut doc = TantivyDocument::default();
            doc.add_bytes(bytes_field, &bytes);
            store_writer.store(&doc, &schema)?;
            docs.push(bytes);
        }
        store_writer.close()?;
    }
    let store = StoreReader::open(directory.open_read(path)?, 0)?;
    assert_eq!(store.decompressor(), Decompressor::Zstd);
    for (doc_id, bytes) in docs.iter().enumerate() {
        let doc: TantivyDocument = store.get(doc_id as u32)?;
        assert_eq!(
            doc.get_first(bytes_field).unwrap().as_bytes(),
            Some(&bytes[..])
        );
    }
    Ok(())
}

#[test]
fn test_store_zstd_dict_invalid_dict_size() {
    let directory = RamDirectory::create();
    for dict_size in [0, ZstdCompressor::MAX_DICT_SIZE + 1] {
        let wrt = directory
            .open_write(Path::new(&format!("store{dict_size}")))
            .unwrap();
        assert!(StoreWriter::new(wrt, zstd_dict(None, dict_size), BLOCK_SIZE, false).is_err());
    }
}

/// Rewrites the footer and the dictionary of a store with a dictionary. The footer records the
/// checksum of `dict`.
fn with_dict_and_footer(
    store_bytes: &OwnedBytes,
    dict: &[u8],
    decompressor_id: u8,
    dict_len: u32,
) -> OwnedBytes {
    let footer = footer(store_bytes);
    let dict_start = footer.offset as usize - footer.dict_len as usize;
    let skip_index = &store_bytes.as_slice()
        [footer.offset as usize..store_bytes.len() - DocStoreFooter::SIZE_IN_BYTES];
    let mut bytes = store_bytes.as_slice()[..dict_start].to_vec();
    bytes.extend_from_slice(dict);
    let new_footer =
        DocStoreFooter::with_dictionary(bytes.len() as u64, dict_len, crc32fast::hash(dict));
    bytes.extend_from_slice(skip_index);
    let footer_start = bytes.len();
    new_footer.serialize(&mut bytes).unwrap();
    bytes[footer_start + 12] = decompressor_id;
    OwnedBytes::new(bytes)
}

#[test]
fn test_store_zstd_dict_corruptions() -> crate::Result<()> {
    let store_bytes = write_store(zstd_dict(None, DICT_SIZE), 1_000, true);
    let dict = embedded_dict(&store_bytes);
    let dict_len = dict.len() as u32;

    // Sanity check: rewriting with the same values yields a valid store.
    let same = with_dict_and_footer(&store_bytes, &dict, 5, dict_len);
    assert_eq!(same.as_slice(), store_bytes.as_slice());

    // Corrupted dictionary content: the checksum does not match.
    let mut bad_content = store_bytes.to_vec();
    let dict_start = footer(&store_bytes).offset as usize - dict.len();
    bad_content[dict_start + dict.len() / 2] ^= 0xFF;
    assert!(StoreReader::open(FileSlice::from(bad_content), 0).is_err());

    // Corrupted magic number.
    let mut bad_magic = dict.clone();
    bad_magic[0] ^= 0xFF;
    let corrupted = with_dict_and_footer(&store_bytes, &bad_magic, 5, dict_len);
    assert!(StoreReader::open(file(&corrupted), 0).is_err());

    // Missing dictionary.
    let missing = with_dict_and_footer(&store_bytes, &[], 5, 0);
    assert!(StoreReader::open(file(&missing), 0).is_err());

    // Dictionary length out of bounds.
    let out_of_bounds = with_dict_and_footer(&store_bytes, &dict, 5, u32::MAX);
    assert!(StoreReader::open(file(&out_of_bounds), 0).is_err());

    // Unknown compressor id: an error, not a panic.
    let unknown = with_dict_and_footer(&store_bytes, &dict, 42, dict_len);
    assert!(StoreReader::open(file(&unknown), 0).is_err());

    // A valid dictionary that is not the one the blocks were compressed with.
    let other_store = write_store(zstd_dict(Some(1), DICT_SIZE / 2), 1_000, true);
    let other_dict = embedded_dict(&other_store);
    assert_ne!(other_dict, dict);
    let wrong_dict = with_dict_and_footer(&store_bytes, &other_dict, 5, other_dict.len() as u32);
    let store = StoreReader::open(file(&wrong_dict), 0)?;
    assert!(store.get::<TantivyDocument>(0).is_err());
    assert!(store.iter::<TantivyDocument>(None).any(|doc| doc.is_err()));
    Ok(())
}

#[test]
fn test_footer_bytes_unchanged_without_dictionary() {
    let footer = DocStoreFooter::new(
        0x0102_0304_0506_0708,
        Decompressor::Zstd,
        super::DOC_STORE_VERSION,
    );
    let mut bytes = Vec::new();
    footer.serialize(&mut bytes).unwrap();
    let mut expected = vec![2, 0, 0, 0, 8, 7, 6, 5, 4, 3, 2, 1, 4];
    expected.extend_from_slice(&[0; 15]);
    assert_eq!(bytes, expected);
    assert_eq!(bytes.len(), DocStoreFooter::SIZE_IN_BYTES);
    // Reserved bytes are ignored for compressors without dictionary.
    bytes[13] = 1;
    let deserialized = DocStoreFooter::deserialize(&mut &bytes[..]).unwrap();
    assert_eq!(deserialized, footer);
}

#[test]
fn test_footer_bytes_with_dictionary() {
    let footer = DocStoreFooter::with_dictionary(0x0102_0304_0506_0708, 0x0A0B_0C0D, 0x1122_3344);
    let mut bytes = Vec::new();
    footer.serialize(&mut bytes).unwrap();
    let mut expected = vec![3, 0, 0, 0, 8, 7, 6, 5, 4, 3, 2, 1, 5];
    expected.extend_from_slice(&[0x0D, 0x0C, 0x0B, 0x0A, 0x44, 0x33, 0x22, 0x11]);
    expected.extend_from_slice(&[0; 7]);
    assert_eq!(bytes, expected);
    assert_eq!(
        DocStoreFooter::deserialize(&mut &bytes[..]).unwrap(),
        footer
    );
    // Older versions of tantivy only know V1 and V2, and reject the store.
    let mut v2 = bytes.clone();
    v2[0] = 2;
    assert!(DocStoreFooter::deserialize(&mut &v2[..]).is_err());
}

#[test]
fn test_store_version_depends_on_dictionary() {
    let with_dict = write_store(zstd_dict(None, DICT_SIZE), 1_000, true);
    assert_eq!(footer(&with_dict).doc_store_version, DocStoreVersion::V3);
    let fallback = write_store(zstd_dict(None, DICT_SIZE), 10, true);
    assert_eq!(footer(&fallback).doc_store_version, DocStoreVersion::V2);
    let plain = write_store(Compressor::Zstd(ZstdCompressor::default()), 1_000, true);
    assert_eq!(footer(&plain).doc_store_version, DocStoreVersion::V2);
}

#[test]
fn test_stack_refuses_dictionaries() -> crate::Result<()> {
    let directory = RamDirectory::create();
    let dict_store = write_store(zstd_dict(None, DICT_SIZE), 1_000, true);
    let plain_store = write_store(Compressor::Zstd(ZstdCompressor::default()), 1_000, true);

    for (i, (reader_bytes, writer_compressor)) in [
        (
            dict_store.clone(),
            Compressor::Zstd(ZstdCompressor::default()),
        ),
        (dict_store, zstd_dict(None, DICT_SIZE)),
        (plain_store, zstd_dict(None, DICT_SIZE)),
    ]
    .into_iter()
    .enumerate()
    {
        for separate_thread in [false, true] {
            let path = format!("stacked{i}{separate_thread}");
            let path = Path::new(&path);
            let mut writer = StoreWriter::new(
                directory.open_write(path)?,
                writer_compressor,
                BLOCK_SIZE,
                separate_thread,
            )?;
            let reader = StoreReader::open(file(&reader_bytes), 1)?;
            let res = writer.stack(reader).and_then(|_| writer.close());
            assert!(res.is_err());
        }
    }
    Ok(())
}

// --- Merges -----------------------------------------------------------------------------

const WORDS: [&str; 16] = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
    "kilo", "lima", "mike", "november", "oscar", "papa",
];

/// Deterministic text for a doc id. Each segment uses its own vocabulary so that the trained
/// dictionaries differ.
fn body_for(id: u64) -> String {
    let segment = id / 10_000;
    let mut state = id.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut body = format!("doc {id}:");
    for _ in 0..30 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        body.push(' ');
        body.push_str(WORDS[(state % 16) as usize]);
        body.push_str(&segment.to_string());
    }
    body
}

fn merge_test_index(
    compressor: Compressor,
) -> crate::Result<(Index, schema::Field, schema::Field)> {
    let mut schema_builder = Schema::builder();
    let id_field = schema_builder.add_u64_field("id", INDEXED | STORED);
    let body_field = schema_builder.add_text_field("body", STORED);
    let index = Index::builder()
        .schema(schema_builder.build())
        .settings(IndexSettings {
            docstore_compression: compressor,
            docstore_blocksize: BLOCK_SIZE,
            ..Default::default()
        })
        .create_in_ram()?;
    Ok((index, id_field, body_field))
}

/// Writes one segment per entry, each with `num_docs` docs of about 300 bytes.
fn add_segments(
    index: &Index,
    id_field: schema::Field,
    body_field: schema::Field,
    segments: std::ops::Range<u64>,
    num_docs: u64,
) -> crate::Result<()> {
    // Big enough for the dictionary sample: segments are only flushed on commit.
    let mut index_writer: IndexWriter = index.writer_with_num_threads(1, 50_000_000)?;
    index_writer.set_merge_policy(Box::new(NoMergePolicy));
    for segment in segments {
        for id in segment * 10_000..segment * 10_000 + num_docs {
            index_writer.add_document(doc!(id_field => id, body_field => body_for(id)))?;
        }
        index_writer.commit()?;
    }
    Ok(())
}

fn store_files(index: &Index) -> crate::Result<Vec<OwnedBytes>> {
    let searcher = index.reader()?.searcher();
    searcher
        .segment_readers()
        .iter()
        .map(|segment_reader| {
            let path = segment_reader.segment_id().uuid_string().to_string() + ".store";
            Ok(index
                .directory()
                .open_read(Path::new(&path))?
                .read_bytes()?)
        })
        .collect()
}

fn merge_all(index: &Index) -> crate::Result<()> {
    let segment_ids = index.searchable_segment_ids()?;
    let mut index_writer: IndexWriter = index.writer_for_tests()?;
    index_writer.merge(&segment_ids).wait()?;
    index_writer.wait_merging_threads()?;
    Ok(())
}

/// Checks that the single segment of the index holds exactly `expected_ids`.
fn check_single_segment(
    index: &Index,
    id_field: schema::Field,
    body_field: schema::Field,
    expected_ids: &[u64],
    expected_decompressor: Decompressor,
) -> crate::Result<()> {
    let searcher = index.reader()?.searcher();
    assert_eq!(searcher.segment_readers().len(), 1);
    let segment_reader = searcher.segment_reader(0);
    let store = segment_reader.get_store_reader(0)?;
    assert_eq!(store.decompressor(), expected_decompressor);
    let mut ids = Vec::new();
    for doc in store.iter::<TantivyDocument>(segment_reader.alive_bitset()) {
        let doc = doc?;
        let id = doc.get_first(id_field).unwrap().as_u64().unwrap();
        assert_eq!(
            doc.get_first(body_field).unwrap().as_str(),
            Some(&*body_for(id))
        );
        ids.push(id);
    }
    ids.sort_unstable();
    assert_eq!(ids, expected_ids);
    // Random access goes through the dictionary too.
    for doc_id in (0..segment_reader.max_doc()).step_by(97) {
        if segment_reader.is_deleted(doc_id) {
            continue;
        }
        let doc: TantivyDocument = store.get(doc_id)?;
        let id = doc.get_first(id_field).unwrap().as_u64().unwrap();
        assert_eq!(
            doc.get_first(body_field).unwrap().as_str(),
            Some(&*body_for(id))
        );
    }
    Ok(())
}

const DOCS_PER_SEGMENT: u64 = 2_000;

fn all_ids(segments: std::ops::Range<u64>) -> Vec<u64> {
    segments
        .flat_map(|segment| segment * 10_000..segment * 10_000 + DOCS_PER_SEGMENT)
        .collect()
}

fn test_merge_segments_with_different_dicts(with_deletes: bool) -> crate::Result<()> {
    let (index, id_field, body_field) = merge_test_index(zstd_dict(None, DICT_SIZE))?;
    add_segments(&index, id_field, body_field, 0..3, DOCS_PER_SEGMENT)?;

    let stores = store_files(&index)?;
    assert_eq!(stores.len(), 3);
    let dicts: Vec<Vec<u8>> = stores.iter().map(embedded_dict).collect();
    for store in &stores {
        assert_eq!(footer(store).decompressor, Decompressor::ZstdDict);
    }
    assert!(dicts[0] != dicts[1] && dicts[1] != dicts[2] && dicts[0] != dicts[2]);

    let mut expected_ids = all_ids(0..3);
    if with_deletes {
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        for &id in expected_ids.iter().filter(|id| *id % 7 == 0) {
            index_writer.delete_term(Term::from_field_u64(id_field, id));
        }
        index_writer.commit()?;
        expected_ids.retain(|id| id % 7 != 0);
    }

    merge_all(&index)?;
    check_single_segment(
        &index,
        id_field,
        body_field,
        &expected_ids,
        Decompressor::ZstdDict,
    )?;
    Ok(())
}

#[test]
fn test_merge_segments_with_different_dicts_no_deletes() -> crate::Result<()> {
    test_merge_segments_with_different_dicts(false)
}

#[test]
fn test_merge_segments_with_different_dicts_with_deletes() -> crate::Result<()> {
    test_merge_segments_with_different_dicts(true)
}

#[test]
fn test_merge_dict_segments_into_plain_zstd() -> crate::Result<()> {
    let (mut index, id_field, body_field) = merge_test_index(zstd_dict(None, DICT_SIZE))?;
    add_segments(&index, id_field, body_field, 0..2, DOCS_PER_SEGMENT)?;
    index.settings_mut().docstore_compression = Compressor::Zstd(ZstdCompressor::default());
    merge_all(&index)?;
    check_single_segment(
        &index,
        id_field,
        body_field,
        &all_ids(0..2),
        Decompressor::Zstd,
    )
}

#[test]
fn test_merge_plain_zstd_segments_into_dict() -> crate::Result<()> {
    let (mut index, id_field, body_field) =
        merge_test_index(Compressor::Zstd(ZstdCompressor::default()))?;
    add_segments(&index, id_field, body_field, 0..2, DOCS_PER_SEGMENT)?;
    // Big enough to be stacked if the target store did not use a dictionary.
    for store in store_files(&index)? {
        let reader = StoreReader::open(file(&store), 0)?;
        assert!(reader.block_checkpoints().count() >= 6);
    }
    index.settings_mut().docstore_compression = zstd_dict(None, DICT_SIZE);
    merge_all(&index)?;
    check_single_segment(
        &index,
        id_field,
        body_field,
        &all_ids(0..2),
        Decompressor::ZstdDict,
    )
}

#[test]
fn test_merge_fallback_segments_into_dict() -> crate::Result<()> {
    // Tiny segments fall back to plain zstd; the merged store is big enough for a dictionary.
    let (index, id_field, body_field) = merge_test_index(zstd_dict(None, DICT_SIZE))?;
    add_segments(&index, id_field, body_field, 0..20, 100)?;
    for store in store_files(&index)? {
        assert_eq!(footer(&store).decompressor, Decompressor::Zstd);
    }
    merge_all(&index)?;
    let expected_ids: Vec<u64> = (0..20)
        .flat_map(|segment| segment * 10_000..segment * 10_000 + 100)
        .collect();
    check_single_segment(
        &index,
        id_field,
        body_field,
        &expected_ids,
        Decompressor::ZstdDict,
    )
}

#[test]
fn test_zstd_dict_settings_serde() {
    let settings = IndexSettings {
        docstore_compression: zstd_dict(Some(3), 65_536),
        ..Default::default()
    };
    let json = serde_json::to_string(&settings).unwrap();
    assert!(json.contains(r#""docstore_compression":"zstd(compression_level=3,dict_size=65536)""#));
    let deserialized: IndexSettings = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, settings);
}

// --- Memory accounting -------------------------------------------------------------------

/// Stores `body_for` documents in all `writers` until `*num_bytes` reaches `until`.
fn store_docs_until(
    writers: &mut [&mut StoreWriter],
    schema: &Schema,
    next_id: &mut u64,
    num_bytes: &mut usize,
    until: usize,
) -> crate::Result<()> {
    let body = schema.get_field("body").unwrap();
    while *num_bytes < until {
        let text = body_for(*next_id);
        *next_id += 1;
        *num_bytes += text.len();
        let mut doc = TantivyDocument::default();
        doc.add_text(body, text);
        for writer in writers.iter_mut() {
            writer.store(&doc, schema)?;
        }
    }
    Ok(())
}

#[test]
fn test_store_writer_mem_usage_accounts_for_dictionary_sample() -> crate::Result<()> {
    // The sample target is 100 * 65_536 bytes; training becomes possible from 10 * 65_536.
    const DICT_SIZE: usize = 65_536;
    // Blocks queued for, or being compressed by, the dedicated thread are not accounted for.
    const NOT_ACCOUNTED: usize = 6 * BLOCK_SIZE;
    let schema = lorem_schema();
    let directory = RamDirectory::create();
    for separate_thread in [false, true] {
        let mut dict_writer = StoreWriter::new(
            directory.open_write(Path::new(&format!("dict{separate_thread}")))?,
            zstd_dict(None, DICT_SIZE),
            BLOCK_SIZE,
            separate_thread,
        )?;
        let mut plain_writer = StoreWriter::new(
            directory.open_write(Path::new(&format!("plain{separate_thread}")))?,
            Compressor::Zstd(ZstdCompressor::default()),
            BLOCK_SIZE,
            separate_thread,
        )?;
        let (mut next_id, mut num_bytes) = (0, 0);
        store_docs_until(
            &mut [&mut dict_writer, &mut plain_writer],
            &schema,
            &mut next_id,
            &mut num_bytes,
            2 << 20,
        )?;
        // The sample, and the memory reserved to train the dictionary on it.
        assert!(
            dict_writer.mem_usage() >= num_bytes - NOT_ACCOUNTED + (10 << 20),
            "{} for {num_bytes} bytes",
            dict_writer.mem_usage()
        );
        assert!(plain_writer.mem_usage() < 4 * BLOCK_SIZE);

        // Once the sample is complete, only the encoder remains.
        store_docs_until(
            &mut [&mut dict_writer, &mut plain_writer],
            &schema,
            &mut next_id,
            &mut num_bytes,
            100 * DICT_SIZE + NOT_ACCOUNTED + 2 * BLOCK_SIZE,
        )?;
        assert!(
            dict_writer.mem_usage() < 2 << 20,
            "{}",
            dict_writer.mem_usage()
        );
        dict_writer.close()?;
        plain_writer.close()?;
        let store_bytes = directory
            .open_read(Path::new(&format!("dict{separate_thread}")))?
            .read_bytes()?;
        assert_eq!(footer(&store_bytes).decompressor, Decompressor::ZstdDict);
    }
    Ok(())
}

#[test]
fn test_memory_budget_accounts_for_dictionary_sample() -> crate::Result<()> {
    let mut num_segments = Vec::new();
    for compressor in [
        Compressor::Zstd(ZstdCompressor::default()),
        zstd_dict(None, 65_536),
    ] {
        let (index, id_field, body_field) = merge_test_index(compressor)?;
        let mut index_writer: IndexWriter = index.writer_with_num_threads(
            1,
            crate::indexer::index_writer::MEMORY_BUDGET_NUM_BYTES_MIN,
        )?;
        index_writer.set_merge_policy(Box::new(NoMergePolicy));
        // About 3 MB of stored text.
        for id in 0..15_000 {
            index_writer.add_document(doc!(id_field => id, body_field => body_for(id)))?;
        }
        index_writer.commit()?;
        num_segments.push(index.searchable_segment_ids()?.len());
    }
    // The 15 MB budget holds the documents and the plain doc store, but not the dictionary
    // sample plus the memory needed to train on it.
    assert_eq!(num_segments[0], 1);
    assert!(num_segments[1] > 1, "{num_segments:?}");
    Ok(())
}
