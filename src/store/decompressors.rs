use std::io;
#[cfg(feature = "zstd-compression")]
use std::sync::Arc;

use common::HasLen;
use serde::{Deserialize, Serialize};

#[cfg(feature = "zstd-compression")]
use super::compression_zstd_block::ZstdDictDecoder;
use super::footer::DocStoreFooter;
use super::Compressor;
use crate::directory::FileSlice;

/// Decompressor is deserialized from the doc store footer, when opening an index.
#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decompressor {
    /// No compression
    None,
    /// Use the lz4 decompressor (block format)
    #[cfg(feature = "lz4-compression")]
    Lz4,
    /// Use the zstd decompressor
    #[cfg(feature = "zstd-compression")]
    Zstd,
    /// Use the zstd decompressor with the dictionary embedded in the doc store.
    #[cfg(feature = "zstd-compression")]
    ZstdDict,
}

/// Returns the decompressor of the blocks written by `compressor`.
///
/// A zstd compressor with `dict_size` maps to `Zstd`: whether a store actually uses a
/// dictionary (`ZstdDict`) is only decided when it is closed.
impl From<Compressor> for Decompressor {
    fn from(compressor: Compressor) -> Self {
        match compressor {
            Compressor::None => Decompressor::None,
            #[cfg(feature = "lz4-compression")]
            Compressor::Lz4 => Decompressor::Lz4,
            #[cfg(feature = "zstd-compression")]
            Compressor::Zstd(_) => Decompressor::Zstd,
        }
    }
}

impl Decompressor {
    pub(crate) fn from_id(id: u8) -> io::Result<Decompressor> {
        match id {
            0 => Ok(Decompressor::None),
            #[cfg(feature = "lz4-compression")]
            1 => Ok(Decompressor::Lz4),
            #[cfg(feature = "zstd-compression")]
            4 => Ok(Decompressor::Zstd),
            #[cfg(feature = "zstd-compression")]
            5 => Ok(Decompressor::ZstdDict),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown compressor id {id:?}"),
            )),
        }
    }

    /// Returns true if blocks can only be decompressed with the dictionary embedded in the store.
    pub(crate) fn has_embedded_dictionary(&self) -> bool {
        match self {
            #[cfg(feature = "zstd-compression")]
            Self::ZstdDict => true,
            _ => false,
        }
    }

    pub(crate) fn get_id(&self) -> u8 {
        match self {
            Self::None => 0,
            #[cfg(feature = "lz4-compression")]
            Self::Lz4 => 1,
            #[cfg(feature = "zstd-compression")]
            Self::Zstd => 4,
            #[cfg(feature = "zstd-compression")]
            Self::ZstdDict => 5,
        }
    }

    pub(crate) fn decompress(&self, compressed_block: &[u8]) -> io::Result<Vec<u8>> {
        let mut decompressed_block = vec![];
        self.decompress_into(compressed_block, &mut decompressed_block)?;
        Ok(decompressed_block)
    }

    #[inline]
    pub(crate) fn decompress_into(
        &self,
        compressed: &[u8],
        decompressed: &mut Vec<u8>,
    ) -> io::Result<()> {
        match self {
            Self::None => {
                decompressed.clear();
                decompressed.extend_from_slice(compressed);
                Ok(())
            }
            #[cfg(feature = "lz4-compression")]
            Self::Lz4 => super::compression_lz4_block::decompress(compressed, decompressed),
            #[cfg(feature = "zstd-compression")]
            Self::Zstd => super::compression_zstd_block::decompress(compressed, decompressed),
            #[cfg(feature = "zstd-compression")]
            Self::ZstdDict => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a zstd dictionary is required to decompress this block",
            )),
        }
    }
}

/// Decompresses the blocks of one doc store: the decompressor and, if any, the dictionary
/// embedded in that store.
#[derive(Clone)]
pub(crate) struct BlockDecompressor {
    decompressor: Decompressor,
    #[cfg(feature = "zstd-compression")]
    zstd_dict: Option<Arc<ZstdDictDecoder>>,
}

impl BlockDecompressor {
    /// `dict_file` holds the dictionary described by the footer (empty without dictionary).
    pub(crate) fn open(footer: &DocStoreFooter, dict_file: FileSlice) -> io::Result<Self> {
        let decompressor = footer.decompressor;
        #[cfg(feature = "zstd-compression")]
        if decompressor.has_embedded_dictionary() {
            let zstd_dict = ZstdDictDecoder::open(dict_file.read_bytes()?, footer.dict_checksum)?;
            return Ok(BlockDecompressor {
                decompressor,
                zstd_dict: Some(Arc::new(zstd_dict)),
            });
        }
        debug_assert!(dict_file.is_empty());
        Ok(BlockDecompressor {
            decompressor,
            #[cfg(feature = "zstd-compression")]
            zstd_dict: None,
        })
    }

    pub(crate) fn decompressor(&self) -> Decompressor {
        self.decompressor
    }

    pub(crate) fn decompress(&self, compressed_block: &[u8]) -> io::Result<Vec<u8>> {
        #[cfg(feature = "zstd-compression")]
        if let Some(zstd_dict) = &self.zstd_dict {
            let mut decompressed_block = Vec::new();
            zstd_dict.decompress(compressed_block, &mut decompressed_block)?;
            return Ok(decompressed_block);
        }
        self.decompressor.decompress(compressed_block)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compressor_decompressor_id_test() {
        assert_eq!(Decompressor::from(Compressor::None), Decompressor::None);
        #[cfg(feature = "lz4-compression")]
        assert_eq!(Decompressor::from(Compressor::Lz4), Decompressor::Lz4);
        #[cfg(feature = "zstd-compression")]
        assert_eq!(
            Decompressor::from(Compressor::Zstd(Default::default())),
            Decompressor::Zstd
        );
    }
}
