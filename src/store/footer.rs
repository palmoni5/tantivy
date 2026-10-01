use std::io;

use common::{BinarySerializable, FixedSize, HasLen};

use super::{Decompressor, DocStoreVersion};
use crate::directory::FileSlice;

#[derive(Debug, Clone, PartialEq)]
pub struct DocStoreFooter {
    pub offset: u64,
    pub doc_store_version: DocStoreVersion,
    pub decompressor: Decompressor,
    /// Length of the zstd dictionary stored right before the skip index.
    /// Always 0 unless the decompressor is `ZstdDict`.
    pub dict_len: u32,
    /// CRC32 of the zstd dictionary. Always 0 unless the decompressor is `ZstdDict`.
    pub dict_checksum: u32,
}

/// Serialises the footer to a byte-array
/// - doc store version: 4 bytes
/// - offset : 8 bytes
/// - compressor id: 1 byte
/// - reserved for future use: 15 bytes.
///
/// Doc stores embedding a zstd dictionary use doc store version `V3` and compressor id 5
/// (`ZstdDict`), and store the dictionary length and CRC32 (u32 LE each) in the first 8 reserved
/// bytes. Older readers reject version `V3`. All other doc stores use version `V2` and write
/// zeros in the reserved bytes, as before.
impl BinarySerializable for DocStoreFooter {
    fn serialize<W: io::Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        BinarySerializable::serialize(&self.doc_store_version, writer)?;
        BinarySerializable::serialize(&self.offset, writer)?;
        BinarySerializable::serialize(&self.decompressor.get_id(), writer)?;
        let mut reserved = [0u8; 15];
        reserved[..4].copy_from_slice(&self.dict_len.to_le_bytes());
        reserved[4..8].copy_from_slice(&self.dict_checksum.to_le_bytes());
        writer.write_all(&reserved)?;
        Ok(())
    }

    fn deserialize<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let doc_store_version = DocStoreVersion::deserialize(reader)?;
        let offset = u64::deserialize(reader)?;
        let compressor_id = u8::deserialize(reader)?;
        let mut reserved = [0; 15];
        reader.read_exact(&mut reserved)?;
        let decompressor = Decompressor::from_id(compressor_id)?;
        let has_dictionary = doc_store_version == DocStoreVersion::V3;
        if decompressor.has_embedded_dictionary() != has_dictionary {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "doc store version {doc_store_version} does not match compressor \
                     {decompressor:?}"
                ),
            ));
        }
        let (dict_len, dict_checksum) = if has_dictionary {
            (
                u32::from_le_bytes(reserved[..4].try_into().unwrap()),
                u32::from_le_bytes(reserved[4..8].try_into().unwrap()),
            )
        } else {
            (0, 0)
        };
        Ok(DocStoreFooter {
            offset,
            doc_store_version,
            decompressor,
            dict_len,
            dict_checksum,
        })
    }
}

impl FixedSize for DocStoreFooter {
    const SIZE_IN_BYTES: usize = 28;
}

impl DocStoreFooter {
    pub fn new(
        offset: u64,
        decompressor: Decompressor,
        doc_store_version: DocStoreVersion,
    ) -> Self {
        DocStoreFooter {
            offset,
            doc_store_version,
            decompressor,
            dict_len: 0,
            dict_checksum: 0,
        }
    }

    /// Footer of a doc store embedding a zstd dictionary of `dict_len` bytes.
    #[cfg(feature = "zstd-compression")]
    pub fn with_dictionary(offset: u64, dict_len: u32, dict_checksum: u32) -> Self {
        DocStoreFooter {
            offset,
            doc_store_version: DocStoreVersion::V3,
            decompressor: Decompressor::ZstdDict,
            dict_len,
            dict_checksum,
        }
    }

    pub fn extract_footer(file: FileSlice) -> io::Result<(DocStoreFooter, FileSlice)> {
        if file.len() < DocStoreFooter::SIZE_IN_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "File corrupted. The file is smaller than Footer::SIZE_IN_BYTES (len={}).",
                    file.len()
                ),
            ));
        }
        let (body, footer_slice) = file.split_from_end(DocStoreFooter::SIZE_IN_BYTES);
        let mut footer_bytes = footer_slice.read_bytes()?;
        let footer = DocStoreFooter::deserialize(&mut footer_bytes)?;
        Ok((footer, body))
    }
}

#[test]
fn doc_store_footer_test() {
    // This test is just to safe guard changes on the footer.
    // When the doc store footer is updated, make sure to update also the serialize/deserialize
    // methods
    assert_eq!(core::mem::size_of::<DocStoreFooter>(), 24);
}

#[test]
fn doc_store_footer_golden_bytes_without_dictionary() {
    let footer = DocStoreFooter::new(
        0x0102_0304_0506_0708,
        Decompressor::None,
        super::DOC_STORE_VERSION,
    );
    let mut bytes = Vec::new();
    footer.serialize(&mut bytes).unwrap();
    let mut expected = vec![2, 0, 0, 0, 8, 7, 6, 5, 4, 3, 2, 1, 0];
    expected.extend_from_slice(&[0; 15]);
    assert_eq!(bytes, expected);
    assert_eq!(bytes.len(), DocStoreFooter::SIZE_IN_BYTES);
    // Reserved bytes are ignored without a dictionary.
    bytes[13..21].fill(0xAB);
    assert_eq!(
        DocStoreFooter::deserialize(&mut &bytes[..]).unwrap(),
        footer
    );
    // A dictionary requires doc store version V3, and V3 requires a dictionary.
    bytes[0] = 3;
    assert!(DocStoreFooter::deserialize(&mut &bytes[..]).is_err());
}
