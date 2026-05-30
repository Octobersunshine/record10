use rand::Rng;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::rc::Rc;

const MAX_LEVEL: usize = 32;
const P: f64 = 0.5;

type NodeRef<K, V> = Rc<RefCell<Node<K, V>>>;

struct Node<K, V> {
    key: K,
    value: V,
    forward: Vec<Option<NodeRef<K, V>>>,
}

impl<K, V> Node<K, V> {
    fn new(key: K, value: V, level: usize) -> Self {
        Node {
            key,
            value,
            forward: vec![None; level + 1],
        }
    }
}

pub struct SkipList<K, V> {
    head: NodeRef<K, V>,
    level: usize,
    len: usize,
    rng: rand::rngs::ThreadRng,
}

impl<K: Ord, V> SkipList<K, V> {
    pub fn new() -> Self
    where
        K: Default,
        V: Default,
    {
        let head = Rc::new(RefCell::new(Node::new(
            K::default(),
            V::default(),
            MAX_LEVEL,
        )));
        SkipList {
            head,
            level: 0,
            len: 0,
            rng: rand::thread_rng(),
        }
    }

    fn random_level(&mut self) -> usize {
        let mut level = 0;
        while self.rng.gen_bool(P) && level < MAX_LEVEL - 1 {
            level += 1;
        }
        level
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let mut update = vec![None; MAX_LEVEL + 1];
        let mut current = Some(self.head.clone());

        for i in (0..=self.level).rev() {
            while let Some(ref node) = current {
                let next = node.borrow().forward[i].clone();
                if let Some(ref next_node) = next {
                    if next_node.borrow().key < key {
                        current = next;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            update[i] = current.clone();
        }

        let current = current.and_then(|node| node.borrow().forward[0].clone());
        if let Some(ref node) = current {
            if node.borrow().key == key {
                let old_value = std::mem::replace(&mut node.borrow_mut().value, value);
                return Some(old_value);
            }
        }

        let new_level = self.random_level();
        if new_level > self.level {
            for i in self.level + 1..=new_level {
                update[i] = Some(self.head.clone());
            }
            self.level = new_level;
        }

        let new_node = Rc::new(RefCell::new(Node::new(key, value, new_level)));

        for i in 0..=new_level {
            if let Some(ref update_node) = update[i] {
                let mut update_borrow = update_node.borrow_mut();
                new_node.borrow_mut().forward[i] = update_borrow.forward[i].clone();
                update_borrow.forward[i] = Some(new_node.clone());
            }
        }

        self.len += 1;
        None
    }

    pub fn delete(&mut self, key: &K) -> Option<V> {
        let mut update = vec![None; MAX_LEVEL + 1];
        let mut current = Some(self.head.clone());

        for i in (0..=self.level).rev() {
            while let Some(ref node) = current {
                let next = node.borrow().forward[i].clone();
                if let Some(ref next_node) = next {
                    if &next_node.borrow().key < key {
                        current = next;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            update[i] = current.clone();
        }

        let current = current.and_then(|node| node.borrow().forward[0].clone());
        if let Some(node) = current {
            if &node.borrow().key == key {
                for i in 0..=self.level {
                    if let Some(ref update_node) = update[i] {
                        let mut update_borrow = update_node.borrow_mut();
                        if update_borrow.forward[i]
                            .as_ref()
                            .map_or(false, |n| Rc::ptr_eq(n, &node))
                        {
                            update_borrow.forward[i] = node.borrow().forward[i].clone();
                        }
                    }
                }

                while self.level > 0
                    && self.head.borrow().forward[self.level].is_none()
                {
                    self.level -= 1;
                }

                self.len -= 1;

                drop(update);

                if let Ok(node) = Rc::try_unwrap(node) {
                    return Some(node.into_inner().value);
                }
            }
        }

        None
    }

    pub fn get(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        let mut current = Some(self.head.clone());

        for i in (0..=self.level).rev() {
            while let Some(ref node) = current {
                let next = node.borrow().forward[i].clone();
                if let Some(ref next_node) = next {
                    match next_node.borrow().key.cmp(key) {
                        Ordering::Less => current = next,
                        Ordering::Equal => {
                            return Some(next_node.borrow().value.clone());
                        }
                        Ordering::Greater => break,
                    }
                } else {
                    break;
                }
            }
        }

        None
    }

    pub fn with_value<F, R>(&self, key: &K, f: F) -> Option<R>
    where
        F: FnOnce(&V) -> R,
    {
        let mut current = Some(self.head.clone());

        for i in (0..=self.level).rev() {
            while let Some(ref node) = current {
                let next = node.borrow().forward[i].clone();
                if let Some(ref next_node) = next {
                    match next_node.borrow().key.cmp(key) {
                        Ordering::Less => current = next,
                        Ordering::Equal => {
                            return Some(f(&next_node.borrow().value));
                        }
                        Ordering::Greater => break,
                    }
                } else {
                    break;
                }
            }
        }

        None
    }

    pub fn search(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        self.get(key)
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.search(key).is_some()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn level_counts(&self) -> Vec<usize> {
        let mut counts = vec![0; self.level + 1];
        let mut current = self.head.borrow().forward[0].clone();

        while let Some(ref node) = current {
            let node_level = node.borrow().forward.len() - 1;
            for i in 0..=node_level {
                if i <= self.level {
                    counts[i] += 1;
                }
            }
            current = node.borrow().forward[0].clone();
        }

        counts
    }

    pub fn structure(&self) -> String {
        let mut result = String::new();
        let counts = self.level_counts();

        result.push_str(&format!("SkipList (levels: {}, total nodes: {})\n", self.level + 1, self.len));

        for i in (0..=self.level).rev() {
            result.push_str(&format!("Level {} ({} nodes): ", i, counts[i]));
            let mut current = self.head.borrow().forward[i].clone();
            let mut keys = Vec::new();
            while let Some(ref node) = current {
                keys.push(format!("{:?}", node.borrow().key));
                current = node.borrow().forward[i].clone();
            }
            result.push_str(&keys.join(" -> "));
            result.push('\n');
        }

        result
    }
}

impl<K: Ord + Default, V: Default> Default for SkipList<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search() {
        let mut list: SkipList<i32, String> = SkipList::new();
        assert!(list.is_empty());

        assert_eq!(list.insert(1, "one".to_string()), None);
        assert_eq!(list.insert(2, "two".to_string()), None);
        assert_eq!(list.insert(3, "three".to_string()), None);

        assert_eq!(list.len(), 3);
        assert!(!list.is_empty());

        assert_eq!(list.search(&1), Some("one".to_string()));
        assert_eq!(list.search(&2), Some("two".to_string()));
        assert_eq!(list.search(&3), Some("three".to_string()));
        assert_eq!(list.search(&4), None);

        assert!(list.contains_key(&1));
        assert!(!list.contains_key(&4));

        assert_eq!(list.with_value(&1, |v| v.len()), Some(3));
        assert_eq!(list.with_value(&2, |v| v.to_uppercase()), Some("TWO".to_string()));
    }

    #[test]
    fn test_insert_duplicate() {
        let mut list: SkipList<i32, String> = SkipList::new();
        assert_eq!(list.insert(1, "one".to_string()), None);
        assert_eq!(
            list.insert(1, "ONE".to_string()),
            Some("one".to_string())
        );
        assert_eq!(list.len(), 1);
        assert_eq!(list.search(&1), Some(&"ONE".to_string()));
    }

    #[test]
    fn test_delete() {
        let mut list: SkipList<i32, String> = SkipList::new();
        list.insert(1, "one".to_string());
        list.insert(2, "two".to_string());
        list.insert(3, "three".to_string());

        assert_eq!(list.delete(&2), Some("two".to_string()));
        assert_eq!(list.len(), 2);
        assert_eq!(list.search(&2), None);
        assert_eq!(list.search(&1), Some(&"one".to_string()));
        assert_eq!(list.search(&3), Some(&"three".to_string()));

        assert_eq!(list.delete(&4), None);
        assert_eq!(list.delete(&1), Some("one".to_string()));
        assert_eq!(list.delete(&3), Some("three".to_string()));
        assert!(list.is_empty());
    }

    #[test]
    fn test_random_level_distribution() {
        let mut list: SkipList<i32, i32> = SkipList::new();
        for i in 0..1000 {
            list.insert(i, i);
        }

        let counts = list.level_counts();
        println!("Level counts: {:?}", counts);
        println!("{}", list.structure());

        assert!(counts[0] == 1000);
        assert!(counts.len() > 1);
    }

    #[test]
    fn test_large_scale_operations() {
        let mut list: SkipList<i32, i32> = SkipList::new();
        let n = 10000;

        for i in 0..n {
            list.insert(i, i * 2);
        }

        assert_eq!(list.len(), n);

        for i in 0..n {
            assert_eq!(list.search(&i), Some(&(i * 2)));
        }

        for i in 0..n / 2 {
            assert_eq!(list.delete(&i), Some(i * 2));
        }

        assert_eq!(list.len(), n / 2);

        for i in 0..n / 2 {
            assert_eq!(list.search(&i), None);
        }

        for i in n / 2..n {
            assert_eq!(list.search(&i), Some(&(i * 2)));
        }
    }

    #[test]
    fn test_level_counts() {
        let mut list: SkipList<i32, String> = SkipList::new();

        for i in 0..10 {
            list.insert(i, format!("value_{}", i));
        }

        let counts = list.level_counts();
        println!("Level counts: {:?}", counts);

        assert_eq!(counts[0], 10);

        for i in 1..counts.len() {
            assert!(counts[i] <= counts[i - 1]);
        }
    }

    #[test]
    fn test_structure_output() {
        let mut list: SkipList<i32, i32> = SkipList::new();
        for i in 0..5 {
            list.insert(i, i);
        }

        let structure = list.structure();
        println!("{}", structure);

        assert!(structure.contains("SkipList"));
        assert!(structure.contains("Level 0"));
        assert!(structure.contains("total nodes: 5"));
    }
}
