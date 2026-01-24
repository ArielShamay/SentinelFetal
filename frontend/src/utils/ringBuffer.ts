/**
 * Circular buffer for efficient fixed-size data storage
 * Used for maintaining chart data without memory growth
 */

export class RingBuffer<T> {
  private buffer: T[]
  private head: number = 0
  private tail: number = 0
  private count: number = 0
  private capacity: number

  constructor(capacity: number) {
    this.capacity = capacity
    this.buffer = new Array(capacity)
  }

  /**
   * Add a single item to the buffer
   */
  push(item: T): void {
    this.buffer[this.tail] = item
    this.tail = (this.tail + 1) % this.capacity

    if (this.count < this.capacity) {
      this.count++
    } else {
      // Buffer is full, overwrite oldest
      this.head = (this.head + 1) % this.capacity
    }
  }

  /**
   * Add multiple items to the buffer
   */
  pushMany(items: T[]): void {
    for (const item of items) {
      this.push(item)
    }
  }

  /**
   * Convert buffer to array (oldest to newest)
   */
  toArray(): T[] {
    const result: T[] = []
    let idx = this.head

    for (let i = 0; i < this.count; i++) {
      result.push(this.buffer[idx])
      idx = (idx + 1) % this.capacity
    }

    return result
  }

  /**
   * Get the latest N items (newest)
   */
  getLatest(n: number): T[] {
    const count = Math.min(n, this.count)
    const result: T[] = []

    // Start from (tail - count), wrapping around
    let idx = (this.tail - count + this.capacity) % this.capacity

    for (let i = 0; i < count; i++) {
      result.push(this.buffer[idx])
      idx = (idx + 1) % this.capacity
    }

    return result
  }

  /**
   * Get the first (oldest) item
   */
  first(): T | undefined {
    if (this.count === 0) return undefined
    return this.buffer[this.head]
  }

  /**
   * Get the last (newest) item
   */
  last(): T | undefined {
    if (this.count === 0) return undefined
    const idx = (this.tail - 1 + this.capacity) % this.capacity
    return this.buffer[idx]
  }

  /**
   * Current number of items
   */
  get length(): number {
    return this.count
  }

  /**
   * Check if buffer is full
   */
  get isFull(): boolean {
    return this.count === this.capacity
  }

  /**
   * Check if buffer is empty
   */
  get isEmpty(): boolean {
    return this.count === 0
  }

  /**
   * Clear all items
   */
  clear(): void {
    this.head = 0
    this.tail = 0
    this.count = 0
  }
}

/**
 * Create a ring buffer pre-filled with values
 */
export function createFilledRingBuffer<T>(capacity: number, fillValue: T): RingBuffer<T> {
  const buffer = new RingBuffer<T>(capacity)
  for (let i = 0; i < capacity; i++) {
    buffer.push(fillValue)
  }
  return buffer
}
