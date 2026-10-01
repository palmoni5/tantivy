use std::io;

use serde::{Deserialize, Deserializer, Serialize};

use super::Decompressor;

/// Compressor can be used on `IndexSettings` to choose
/// the compressor used to compress the doc store.
///
/// The default is Lz4Block, but also depends on the enabled feature flags.
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum Compressor {
    /// No compression
    None,
    /// Use the lz4 compressor (block format)
    #[cfg(feature = "lz4-compression")]
    Lz4,
    /// Use the zstd compressor
    #[cfg(feature = "zstd-compression")]
    Zstd(ZstdCompressor),
}

impl Serialize for Compressor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        match *self {
            Compressor::None => serializer.serialize_str("none"),
            #[cfg(feature = "lz4-compression")]
            Compressor::Lz4 => serializer.serialize_str("lz4"),
            #[cfg(feature = "zstd-compression")]
            Compressor::Zstd(zstd) => serializer.serialize_str(&zstd.ser_to_string()),
        }
    }
}

impl<'de> Deserialize<'de> for Compressor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {
        let buf = String::deserialize(deserializer)?;
        let compressor = match buf.as_str() {
            "none" => Compressor::None,
            #[cfg(feature = "lz4-compression")]
            "lz4" => Compressor::Lz4,
            #[cfg(not(feature = "lz4-compression"))]
            "lz4" => {
                return Err(serde::de::Error::custom(
                    "unsupported variant `lz4`, please enable Tantivy's `lz4-compression` feature",
                ))
            }
            #[cfg(feature = "zstd-compression")]
            _ if buf.starts_with("zstd") => Compressor::Zstd(
                ZstdCompressor::deser_from_str(&buf).map_err(serde::de::Error::custom)?,
            ),
            #[cfg(not(feature = "zstd-compression"))]
            _ if buf.starts_with("zstd") => {
                return Err(serde::de::Error::custom(
                    "unsupported variant `zstd`, please enable Tantivy's `zstd-compression` \
                     feature",
                ))
            }
            _ => {
                return Err(serde::de::Error::unknown_variant(
                    &buf,
                    &[
                        "none",
                        #[cfg(feature = "lz4-compression")]
                        "lz4",
                        #[cfg(feature = "zstd-compression")]
                        "zstd",
                        #[cfg(feature = "zstd-compression")]
                        "zstd(compression_level=5)",
                    ],
                ));
            }
        };

        Ok(compressor)
    }
}

#[derive(Clone, Default, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// The Zstd compressor, with optional compression level and dictionary size.
///
/// String form (as used in `meta.json`): `zstd`, `zstd(compression_level=5)`,
/// `zstd(dict_size=65536)` or `zstd(compression_level=5,dict_size=65536)`.
pub struct ZstdCompressor {
    /// The compression level, if unset defaults to zstd::DEFAULT_COMPRESSION_LEVEL = 3
    pub compression_level: Option<i32>,
    /// Maximum size in bytes of a zstd dictionary trained for each doc store.
    ///
    /// When set, every doc store trains a dictionary on its first blocks, embeds it in the
    /// `.store` file and compresses all its blocks with it. This mostly helps small blocks and
    /// small documents. Stores that are too small, or for which the dictionary does not help,
    /// silently fall back to plain zstd blocks.
    ///
    /// Doc stores compressed with a dictionary cannot be stacked during merges: their documents
    /// are recompressed. Readers older than the version introducing this option cannot open
    /// such stores.
    ///
    /// Must be within `[ZstdCompressor::MIN_DICT_SIZE, ZstdCompressor::MAX_DICT_SIZE]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dict_size: Option<usize>,
}

#[cfg(feature = "zstd-compression")]
impl ZstdCompressor {
    /// Smallest accepted value for [`ZstdCompressor::dict_size`].
    pub const MIN_DICT_SIZE: usize = 256;
    /// Largest accepted value for [`ZstdCompressor::dict_size`].
    pub const MAX_DICT_SIZE: usize = 1 << 20;

    pub(crate) fn validate(&self) -> Result<(), String> {
        if let Some(dict_size) = self.dict_size {
            if !(Self::MIN_DICT_SIZE..=Self::MAX_DICT_SIZE).contains(&dict_size) {
                return Err(format!(
                    "zstd dict_size must be within [{}, {}], got {dict_size}",
                    Self::MIN_DICT_SIZE,
                    Self::MAX_DICT_SIZE
                ));
            }
        }
        Ok(())
    }

    fn deser_from_str(val: &str) -> Result<ZstdCompressor, String> {
        if !val.starts_with("zstd") {
            return Err(format!("needs to start with zstd, but got {val}"));
        }
        if val == "zstd" {
            return Ok(ZstdCompressor::default());
        }
        let options = val
            .strip_prefix("zstd(")
            .and_then(|options| options.strip_suffix(')'))
            .ok_or_else(|| format!("expected zstd or zstd(<options>), but got {val:?}"))?;

        let mut compressor = ZstdCompressor::default();
        for option in options.split(',') {
            let (opt_name, value) = option
                .split_once('=')
                .ok_or_else(|| format!("no '=' found in option {option:?}"))?;

            match opt_name {
                "compression_level" => {
                    let value = value.parse::<i32>().map_err(|err| {
                        format!("Could not parse value {value} of option {opt_name}, e: {err}")
                    })?;
                    if value >= 15 {
                        warn!(
                            "High zstd compression level detected: {:?}. High compression levels \
                             (>=15) are slow and will limit indexing speed.",
                            value
                        )
                    }
                    compressor.compression_level = Some(value);
                }
                "dict_size" => {
                    let value = value.parse::<usize>().map_err(|err| {
                        format!("Could not parse value {value} of option {opt_name}, e: {err}")
                    })?;
                    compressor.dict_size = Some(value);
                }
                _ => {
                    return Err(format!("unknown zstd option {opt_name:?}"));
                }
            }
        }
        compressor.validate()?;
        Ok(compressor)
    }
    fn ser_to_string(&self) -> String {
        let mut options = Vec::new();
        if let Some(compression_level) = self.compression_level {
            options.push(format!("compression_level={compression_level}"));
        }
        if let Some(dict_size) = self.dict_size {
            options.push(format!("dict_size={dict_size}"));
        }
        if options.is_empty() {
            "zstd".to_string()
        } else {
            format!("zstd({})", options.join(","))
        }
    }
}

impl Default for Compressor {
    #[allow(unreachable_code)]
    fn default() -> Self {
        #[cfg(feature = "lz4-compression")]
        return Compressor::Lz4;

        #[cfg(feature = "zstd-compression")]
        return Compressor::Zstd(ZstdCompressor::default());

        Compressor::None
    }
}

impl Compressor {
    /// Returns true if the raw blocks of a doc store using `decompressor` can be copied as is
    /// into a doc store written with this compressor.
    ///
    /// Dictionaries are trained per store, so blocks are never stacked into or out of a store
    /// using a dictionary.
    pub(crate) fn can_stack(&self, decompressor: Decompressor) -> bool {
        !self.uses_dictionary()
            && !decompressor.has_embedded_dictionary()
            && decompressor == Decompressor::from(*self)
    }

    /// Returns true if doc stores written with this compressor embed a trained dictionary.
    pub(crate) fn uses_dictionary(&self) -> bool {
        match self {
            #[cfg(feature = "zstd-compression")]
            Self::Zstd(zstd_compressor) => zstd_compressor.dict_size.is_some(),
            _ => false,
        }
    }

    #[inline]
    pub(crate) fn compress_into(
        &self,
        uncompressed: &[u8],
        compressed: &mut Vec<u8>,
    ) -> io::Result<()> {
        match self {
            Self::None => {
                compressed.clear();
                compressed.extend_from_slice(uncompressed);
                Ok(())
            }
            #[cfg(feature = "lz4-compression")]
            Self::Lz4 => super::compression_lz4_block::compress(uncompressed, compressed),
            #[cfg(feature = "zstd-compression")]
            Self::Zstd(_zstd_compressor) => super::compression_zstd_block::compress(
                uncompressed,
                compressed,
                _zstd_compressor.compression_level,
            ),
        }
    }
}

#[cfg(all(feature = "zstd-compression", test))]
mod tests {
    use super::*;

    #[test]
    fn zstd_serde_roundtrip() {
        let compressor = ZstdCompressor {
            compression_level: Some(15),
            dict_size: None,
        };

        assert_eq!(
            ZstdCompressor::deser_from_str(&compressor.ser_to_string()).unwrap(),
            compressor
        );

        assert_eq!(
            ZstdCompressor::deser_from_str(&ZstdCompressor::default().ser_to_string()).unwrap(),
            ZstdCompressor::default()
        );
    }

    #[test]
    fn deser_zstd_test() {
        assert_eq!(
            ZstdCompressor::deser_from_str("zstd").unwrap(),
            ZstdCompressor::default()
        );

        assert!(ZstdCompressor::deser_from_str("zzstd").is_err());
        assert!(ZstdCompressor::deser_from_str("zzstd()").is_err());
        assert_eq!(
            ZstdCompressor::deser_from_str("zstd(compression_level=15)").unwrap(),
            ZstdCompressor {
                compression_level: Some(15),
                dict_size: None,
            }
        );
        assert_eq!(
            ZstdCompressor::deser_from_str("zstd(compresion_level=15)").unwrap_err(),
            "unknown zstd option \"compresion_level\""
        );
        assert_eq!(
            ZstdCompressor::deser_from_str("zstd(compression_level->2)").unwrap_err(),
            "no '=' found in option \"compression_level->2\""
        );
        assert_eq!(
            ZstdCompressor::deser_from_str("zstd(compression_level=over9000)").unwrap_err(),
            "Could not parse value over9000 of option compression_level, e: invalid digit found \
             in string"
        );
    }

    #[test]
    fn deser_zstd_each_option_is_parsed_separately() {
        assert_eq!(
            ZstdCompressor::deser_from_str("zstd(compression_level=3,foo=1)").unwrap_err(),
            "unknown zstd option \"foo\""
        );
        assert_eq!(
            ZstdCompressor::deser_from_str("zstd(compression_level=3,foo)").unwrap_err(),
            "no '=' found in option \"foo\""
        );
    }

    #[test]
    fn deser_zstd_malformed_is_an_error() {
        for malformed in [
            "zstd5",
            "zstd(",
            "zstd)",
            "zstdé",
            "zstd(compression_level=",
            "zstd(compression_level=)",
            "zstd(compression_level=35",
            "zstd_compression_level=3)",
        ] {
            assert!(
                ZstdCompressor::deser_from_str(malformed).is_err(),
                "{malformed:?} should be rejected"
            );
            assert!(
                serde_json::from_str::<Compressor>(&format!("{malformed:?}")).is_err(),
                "{malformed:?} should be rejected"
            );
        }
    }

    #[test]
    fn zstd_dict_size_serde() {
        let both = ZstdCompressor {
            compression_level: Some(3),
            dict_size: Some(65_536),
        };
        assert_eq!(
            both.ser_to_string(),
            "zstd(compression_level=3,dict_size=65536)"
        );
        assert_eq!(
            ZstdCompressor::deser_from_str("zstd(compression_level=3,dict_size=65536)").unwrap(),
            both
        );
        assert_eq!(
            ZstdCompressor::deser_from_str("zstd(dict_size=65536,compression_level=3)").unwrap(),
            both
        );
        let dict_only = ZstdCompressor {
            compression_level: None,
            dict_size: Some(4096),
        };
        assert_eq!(dict_only.ser_to_string(), "zstd(dict_size=4096)");
        assert_eq!(
            ZstdCompressor::deser_from_str(&dict_only.ser_to_string()).unwrap(),
            dict_only
        );

        let compressor: Compressor =
            serde_json::from_str(r#""zstd(compression_level=3,dict_size=65536)""#).unwrap();
        assert_eq!(compressor, Compressor::Zstd(both));
        assert_eq!(
            serde_json::to_string(&compressor).unwrap(),
            r#""zstd(compression_level=3,dict_size=65536)""#
        );
    }

    #[test]
    fn zstd_dict_size_deser_errors() {
        assert_eq!(
            ZstdCompressor::deser_from_str("zstd(dict_size=100)").unwrap_err(),
            "zstd dict_size must be within [256, 1048576], got 100"
        );
        assert!(ZstdCompressor::deser_from_str("zstd(dict_size=1048577)").is_err());
        assert_eq!(
            ZstdCompressor::deser_from_str("zstd(dict_size=-1)").unwrap_err(),
            "Could not parse value -1 of option dict_size, e: invalid digit found in string"
        );
        assert_eq!(
            ZstdCompressor::deser_from_str("zstd(compression_level=3,)").unwrap_err(),
            "no '=' found in option \"\""
        );
    }
}
