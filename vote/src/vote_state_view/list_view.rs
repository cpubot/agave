use super::field_frames::ListFrame;

pub(super) struct ListView<'a, F> {
    frame: F,
    item_buffer: &'a [u8],
}

impl<'a, F: ListFrame> ListView<'a, F> {
    pub(super) fn new(frame: F, buffer: &'a [u8]) -> Self {
        let len_offset = core::mem::size_of::<u64>();
        let item_buffer = &buffer[len_offset..];
        Self { frame, item_buffer }
    }

    #[inline]
    pub(super) fn len(&self) -> usize {
        self.frame.len()
    }

    pub(super) fn into_iter(self) -> ListViewIter<'a, F>
    where
        Self: Sized,
    {
        ListViewIter {
            index: 0,
            rev_index: 0,
            view: self,
        }
    }

    pub(super) fn last(&self) -> Option<&F::Item> {
        let len = self.len();
        if len == 0 {
            return None;
        }
        self.item(len - 1)
    }

    fn item(&self, index: usize) -> Option<&'a F::Item> {
        if index >= self.len() {
            return None;
        }

        let offset = index * self.frame.item_size();
        // SAFETY: `item_buffer` is long enough to contain all items
        let item_data = &self.item_buffer[offset..offset + self.frame.item_size()];
        // SAFETY: `item_data` is long enough to contain an item
        Some(unsafe { self.frame.read_item(item_data) })
    }
}

pub(super) struct ListViewIter<'a, F> {
    index: usize,
    rev_index: usize,
    view: ListView<'a, F>,
}

impl<'a, F: ListFrame> Iterator for ListViewIter<'a, F>
where
    F::Item: 'a,
{
    type Item = &'a F::Item;

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.view.len().saturating_sub(self.index + self.rev_index);
        (remaining, Some(remaining))
    }

    fn next(&mut self) -> Option<Self::Item> {
        if self.index + self.rev_index < self.view.len() {
            let item = self.view.item(self.index);
            self.index += 1;
            item
        } else {
            None
        }
    }
}

impl<'a, F: ListFrame> DoubleEndedIterator for ListViewIter<'a, F>
where
    F::Item: 'a,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index + self.rev_index < self.view.len() {
            let item = self.view.item(self.view.len() - self.rev_index - 1);
            self.rev_index += 1;
            item
        } else {
            None
        }
    }
}

impl<'a, F: ListFrame> ExactSizeIterator for ListViewIter<'a, F>
where
    F::Item: 'a,
{
    fn len(&self) -> usize {
        self.view.len().saturating_sub(self.index + self.rev_index)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        super::field_frames::{LandedVotesListFrame, LockoutItem, LockoutListFrame},
        *,
    };

    fn build_lockout_buffer(len: u8) -> Vec<u8> {
        let frame = LockoutListFrame { len };
        let len_offset = core::mem::size_of::<u64>();
        let item_size = frame.item_size();
        let total_size = len_offset + frame.total_item_size();

        let mut buffer = vec![0u8; total_size];
        let len_le = (len as u64).to_le_bytes();
        buffer[..len_offset].copy_from_slice(&len_le);

        for i in 0..(len as usize) {
            let base = len_offset + i * item_size;
            // slot: [u8; 8]
            let slot_le = (i as u64).to_le_bytes();
            buffer[base..base + 8].copy_from_slice(&slot_le);
            // confirmation_count: [u8; 4]
            let conf_le = 0u32.to_le_bytes();
            buffer[base + 8..base + 12].copy_from_slice(&conf_le);
        }

        buffer
    }

    fn build_landed_buffer(len: u8) -> Vec<u8> {
        let frame = LandedVotesListFrame { len };
        let len_offset = core::mem::size_of::<u64>();
        let item_size = frame.item_size();
        let total_size = len_offset + frame.total_item_size();

        let mut buffer = vec![0u8; total_size];
        let len_le = (len as u64).to_le_bytes();
        buffer[..len_offset].copy_from_slice(&len_le);

        for i in 0..(len as usize) {
            let base = len_offset + i * item_size;
            buffer[base] = 7u8;
            let slot_le = (i as u64).to_le_bytes();
            buffer[base + 1..base + 1 + 8].copy_from_slice(&slot_le);
            let conf_le = 0u32.to_le_bytes();
            buffer[base + 1 + 8..base + 1 + 12].copy_from_slice(&conf_le);
        }

        buffer
    }

    #[test]
    fn test_list_view_iter_len_and_no_overlap() {
        let len: u8 = 5;
        let frame = LockoutListFrame { len };
        let buffer = build_lockout_buffer(len);
        let view: ListView<'_, LockoutListFrame> = ListView::new(frame, &buffer);

        let mut iter = view.into_iter();

        // Initial length
        assert_eq!(iter.len(), 5);

        // Consume from front
        let item: &LockoutItem = iter.next().unwrap();
        assert_eq!(item.slot(), 0);
        assert_eq!(iter.len(), 4);

        // Consume from back
        let item = iter.next_back().unwrap();
        assert_eq!(item.slot(), 4);
        assert_eq!(iter.len(), 3);

        let item = iter.next().unwrap();
        assert_eq!(item.slot(), 1);
        assert_eq!(iter.len(), 2);

        let item = iter.next_back().unwrap();
        assert_eq!(item.slot(), 3);
        assert_eq!(iter.len(), 1);

        let item = iter.next().unwrap();
        assert_eq!(item.slot(), 2);
        assert_eq!(iter.len(), 0);

        // Fully exhausted
        assert!(iter.next().is_none());
        assert!(iter.next_back().is_none());
    }

    #[test]
    fn test_size_hint_empty() {
        let frame = LockoutListFrame { len: 0 };
        let buffer = build_lockout_buffer(0);
        let iter = ListView::new(frame, &buffer).into_iter();
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    #[test]
    fn test_size_hint_progress() {
        let frame = LockoutListFrame { len: 3 };
        let buffer = build_lockout_buffer(3);
        let mut iter = ListView::new(frame, &buffer).into_iter();
        assert_eq!(iter.size_hint(), (3, Some(3)));
        let _ = iter.next();
        assert_eq!(iter.size_hint(), (2, Some(2)));
        let _ = iter.next_back();
        assert_eq!(iter.size_hint(), (1, Some(1)));
        let _ = iter.next();
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    #[test]
    fn test_all_front_consumption() {
        let frame = LockoutListFrame { len: 4 };
        let buffer = build_lockout_buffer(4);
        let mut iter = ListView::new(frame, &buffer).into_iter();
        for expected in 0..4u64 {
            assert_eq!(iter.len(), (4 - expected as usize));
            let item = iter.next().unwrap();
            assert_eq!(item.slot(), expected);
        }
        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
        assert!(iter.next_back().is_none());
    }

    #[test]
    fn test_all_back_consumption() {
        let frame = LockoutListFrame { len: 4 };
        let buffer = build_lockout_buffer(4);
        let mut iter = ListView::new(frame, &buffer).into_iter();
        for expected in (0..4u64).rev() {
            assert_eq!(iter.len(), (expected as usize + 1));
            let item = iter.next_back().unwrap();
            assert_eq!(item.slot(), expected);
        }
        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
        assert!(iter.next_back().is_none());
    }

    #[test]
    fn test_last_method() {
        let frame = LockoutListFrame { len: 0 };
        let buffer = build_lockout_buffer(0);
        let view: ListView<'_, LockoutListFrame> = ListView::new(frame, &buffer);
        assert!(view.last().is_none());

        let frame = LockoutListFrame { len: 3 };
        let buffer = build_lockout_buffer(3);
        let view: ListView<'_, LockoutListFrame> = ListView::new(frame, &buffer);
        assert_eq!(view.last().unwrap().slot(), 2);
    }

    #[test]
    fn test_landed_frame_iteration_and_layout() {
        let frame = LandedVotesListFrame { len: 3 };
        let buffer = build_landed_buffer(3);
        let mut iter = ListView::new(frame, &buffer).into_iter();

        assert_eq!(iter.next().unwrap().slot(), 0);
        assert_eq!(iter.next_back().unwrap().slot(), 2);
        assert_eq!(iter.next().unwrap().slot(), 1);
        assert!(iter.next().is_none());
        assert!(iter.next_back().is_none());
    }
}
