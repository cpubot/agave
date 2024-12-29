use {
    serde::{de::DeserializeOwned, Serialize},
    thiserror::Error,
};

#[derive(Error, Debug)]
pub enum Error {
    #[error("not enough bytes to deserialize")]
    NotEnoughBytes,
    #[error("too many bytes to deserialize")]
    TooManyBytes,
    #[error("invalid length")]
    InvalidLength,
    #[error("bincode serialize error")]
    BincodeSerialize,
    #[error("bincode deserialize error")]
    BincodeDeserialize,
}

pub type Result<T> = std::result::Result<T, Error>;

pub trait ToFromBytes {
    fn to_bytes<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&[u8]) -> R;

    fn from_bytes(bytes: &[u8]) -> Result<Self>
    where
        Self: Sized;
}

pub(crate) mod bincode_compat;
pub(crate) mod cursor;
