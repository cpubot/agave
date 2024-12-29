use super::{Error, Result};

pub struct SliceCursor<'a, T> {
    inner: &'a [T],
    pos: usize,
}

impl<'a, T> SliceCursor<'a, T> {
    pub fn new(inner: &'a [T]) -> Self {
        Self { inner, pos: 0 }
    }

    #[inline(always)]
    pub fn read_exact(&mut self, len: usize) -> Result<&'a [T]> {
        let slice = self
            .inner
            .get(self.pos..self.pos + len)
            .ok_or(Error::NotEnoughBytes)?;
        self.pos += len;
        Ok(slice)
    }
}

impl SliceCursor<'_, u8> {
    #[inline(always)]
    pub fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(unsafe {
            self.read_exact(std::mem::size_of::<u64>())?
                .try_into()
                .unwrap_unchecked()
        }))
    }

    #[inline(always)]
    pub fn read_usize(&mut self) -> Result<usize> {
        Ok(usize::from_le_bytes(unsafe {
            self.read_exact(std::mem::size_of::<usize>())?
                .try_into()
                .unwrap_unchecked()
        }))
    }
}
