# EventBus Test Plan — v1.0 Coverage Improvement

> **Current Coverage**: 59% (44/107 lines missing)
> **Target Coverage**: 85%+
> **Priority**: P0 (critical — central nervous system)

---

## Missing Line Analysis

### 1. Input Validation (Lines 85, 87, 89, 98-106)

```python
# Line 85: TypeError for non-callable sync handler
if not callable(callback):
    raise TypeError(f"Handler must be callable, got {type(callback).__name__}")

# Line 87: TypeError for non-string topic
if not isinstance(topic, str):
    raise TypeError(f"Topic must be str, got {type(topic).__name__}")

# Line 89: ValueError for empty topic
if not topic:
    raise ValueError("Topic cannot be empty")

# Lines 98-106: All subscribe_async validation
if not callable(callback):
    raise TypeError(...)
if not isinstance(topic, str):
    raise TypeError(...)
if not topic:
    raise ValueError(...)
if topic not in self._async_subscribers:
    self._async_subscribers[topic] = []
self._async_subscribers[topic].append(callback)
```

**Tests Needed**:
```python
def test_subscribe_rejects_non_callable():
    bus = EventBus()
    with pytest.raises(TypeError, match="Handler must be callable"):
        bus.subscribe("topic", "not_a_function")

def test_subscribe_rejects_non_string_topic():
    bus = EventBus()
    with pytest.raises(TypeError, match="Topic must be str"):
        bus.subscribe(123, lambda e: None)

def test_subscribe_rejects_empty_topic():
    bus = EventBus()
    with pytest.raises(ValueError, match="Topic cannot be empty"):
        bus.subscribe("", lambda e: None)

def test_subscribe_async_rejects_non_callable():
    bus = EventBus()
    with pytest.raises(TypeError):
        bus.subscribe_async("topic", "not_a_function")

def test_subscribe_async_rejects_non_string_topic():
    bus = EventBus()
    with pytest.raises(TypeError):
        bus.subscribe_async(123, lambda e: None)

def test_subscribe_async_rejects_empty_topic():
    bus = EventBus()
    with pytest.raises(ValueError):
        bus.subscribe_async("", lambda e: None)

def test_subscribe_async_creates_new_topic():
    bus = EventBus()
    async def handler(e): pass
    bus.subscribe_async("new.topic", handler)
    assert "new.topic" in bus.topics
```

**Lines covered**: 85, 87, 89, 98-106 (11 lines)

---

### 2. Unsubscribe Async (Line 113)

```python
# Line 113: async callback removal
if topic in self._async_subscribers and callback in self._async_subscribers[topic]:
    self._async_subscribers[topic].remove(callback)
```

**Test Needed**:
```python
def test_unsubscribe_async_callback():
    bus = EventBus()
    async def handler(e): pass
    bus.subscribe_async("topic", handler)
    assert bus.subscriber_count("topic") == 1
    bus.unsubscribe("topic", handler)
    assert bus.subscriber_count("topic") == 0
```

**Lines covered**: 113 (1 line)

---

### 3. History Overflow (Line 125)

```python
# Line 125: pop(0) eviction
if len(self._event_history) > self._max_history:
    self._event_history.pop(0)
```

**Test Needed**:
```python
def test_history_overflow_eviction():
    bus = EventBus()
    bus._max_history = 100  # Lower threshold
    for i in range(150):
        bus.publish(Event(topic="test", payload=i))
    assert len(bus.get_history()) == 100
    assert bus.get_history()[0].payload == 50  # First 50 evicted
```

**Lines covered**: 125 (1 line)

---

### 4. Subscriber Exception Handling (Lines 131-132, 136-139)

```python
# Lines 131-132: sync subscriber exception
except Exception as e:
    print(f"[EventBus] Error in sync subscriber for {event.topic}: {e}")

# Lines 136-139: async subscriber scheduling error
for callback in self._async_subscribers.get(event.topic, []):
    try:
        asyncio.create_task(self._run_async_callback(callback, event))
    except Exception as e:
        print(f"[EventBus] Error scheduling async subscriber for {event.topic}: {e}")
```

**Tests Needed**:
```python
def test_sync_subscriber_exception_doesnt_break_others():
    bus = EventBus()
    received = []
    
    def bad_handler(e):
        raise ValueError("Intentional error")
    
    def good_handler(e):
        received.append(e.payload)
    
    bus.subscribe("test", bad_handler)
    bus.subscribe("test", good_handler)
    
    # Should not raise, good_handler should still run
    bus.publish(Event(topic="test", payload="hello"))
    assert received == ["hello"]

@pytest.mark.asyncio
async def test_async_subscriber_exception_handled():
    bus = EventBus()
    
    async def bad_handler(e):
        raise ValueError("Intentional error")
    
    bus.subscribe_async("test", bad_handler)
    bus.publish(Event(topic="test", payload="hello"))
    
    # Give async task time to run
    await asyncio.sleep(0.1)
    # Should not crash the event loop
```

**Lines covered**: 131-132 (2 lines). Lines 136-139 are hard to trigger — `asyncio.create_task` rarely fails unless event loop is closed.

---

### 5. Async Callback Execution (Lines 145-148)

```python
# Lines 145-148: _run_async_callback
async def _run_async_callback(self, callback, event):
    try:
        await callback(event)
    except Exception as e:
        print(f"[EventBus] Error in async subscriber for {event.topic}: {e}")
```

**Tests Needed**:
```python
@pytest.mark.asyncio
async def test_async_callback_executes():
    bus = EventBus()
    received = []
    
    async def handler(e):
        received.append(e.payload)
    
    bus.subscribe_async("test", handler)
    bus.publish(Event(topic="test", payload="async_hello"))
    
    await asyncio.sleep(0.1)
    assert received == ["async_hello"]

@pytest.mark.asyncio
async def test_async_callback_exception_caught():
    bus = EventBus()
    
    async def bad_handler(e):
        raise RuntimeError("Async error")
    
    bus.subscribe_async("test", bad_handler)
    bus.publish(Event(topic="test", payload="data"))
    
    await asyncio.sleep(0.1)
    # Should not propagate exception
```

**Lines covered**: 145-148 (4 lines)

---

### 6. publish_async (Line 152)

```python
# Line 152: async publish
async def publish_async(self, event: Event) -> None:
    self.publish(event)
```

**Test Needed**:
```python
@pytest.mark.asyncio
async def test_publish_async():
    bus = EventBus()
    received = []
    bus.subscribe("test", lambda e: received.append(e.payload))
    
    await bus.publish_async(Event(topic="test", payload="async_pub"))
    assert received == ["async_pub"]
```

**Lines covered**: 152 (1 line)

---

### 7. clear_history with Topic Filter (Lines 163-166)

```python
# Lines 163-166: filtered clear
if topic:
    self._event_history = [e for e in self._event_history if e.topic != topic]
else:
    self._event_history.clear()
```

**Tests Needed**:
```python
def test_clear_history_specific_topic():
    bus = EventBus()
    bus.publish(Event(topic="a", payload=1))
    bus.publish(Event(topic="b", payload=2))
    bus.publish(Event(topic="a", payload=3))
    
    bus.clear_history(topic="a")
    
    history = bus.get_history()
    assert len(history) == 1
    assert history[0].topic == "b"

def test_clear_all_history():
    bus = EventBus()
    bus.publish(Event(topic="a", payload=1))
    bus.publish(Event(topic="b", payload=2))
    
    bus.clear_history()
    assert len(bus.get_history()) == 0
```

**Lines covered**: 163-166 (4 lines)

---

### 8. subscriber_count (Lines 175-177)

```python
# Lines 175-177: subscriber_count
def subscriber_count(self, topic: str) -> int:
    sync = len(self._subscribers.get(topic, []))
    async_ = len(self._async_subscribers.get(topic, []))
    return sync + async_
```

**Test Needed**:
```python
def test_subscriber_count():
    bus = EventBus()
    bus.subscribe("test", lambda e: None)
    bus.subscribe("test", lambda e: None)
    bus.subscribe_async("test", lambda e: None)
    
    assert bus.subscriber_count("test") == 3
    assert bus.subscriber_count("nonexistent") == 0
```

**Lines covered**: 175-177 (3 lines)

---

### 9. await_event (Lines 199-213) — ENTIRE METHOD UNTESTED

```python
# Lines 199-213: await_event
async def await_event(self, topic, timeout=30.0, filter_fn=None):
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def handler(event):
        if not future.done():
            if filter_fn is None or filter_fn(event):
                future.set_result(event)

    self.subscribe(topic, handler)
    try:
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        return None
    finally:
        self.unsubscribe(topic, handler)
```

**Tests Needed**:
```python
@pytest.mark.asyncio
async def test_await_event_receives_event():
    bus = EventBus()
    
    async def publisher():
        await asyncio.sleep(0.1)
        bus.publish(Event(topic="test", payload="arrived"))
    
    asyncio.create_task(publisher())
    event = await bus.await_event("test", timeout=5.0)
    
    assert event is not None
    assert event.payload == "arrived"

@pytest.mark.asyncio
async def test_await_event_timeout():
    bus = EventBus()
    event = await bus.await_event("test", timeout=0.5)
    assert event is None

@pytest.mark.asyncio
async def test_await_event_with_filter():
    bus = EventBus()
    
    async def publisher():
        await asyncio.sleep(0.05)
        bus.publish(Event(topic="test", payload="skip"))
        await asyncio.sleep(0.05)
        bus.publish(Event(topic="test", payload="match"))
    
    asyncio.create_task(publisher())
    event = await bus.await_event(
        "test",
        timeout=5.0,
        filter_fn=lambda e: e.payload == "match"
    )
    
    assert event is not None
    assert event.payload == "match"

@pytest.mark.asyncio
async def test_await_event_unsubscribes_on_timeout():
    bus = EventBus()
    initial_count = bus.subscriber_count("test")
    
    await bus.await_event("test", timeout=0.1)
    
    # Handler should be cleaned up
    assert bus.subscriber_count("test") == initial_count

@pytest.mark.asyncio
async def test_await_event_unsubscribes_after_receive():
    bus = EventBus()
    
    async def publisher():
        await asyncio.sleep(0.05)
        bus.publish(Event(topic="test", payload="data"))
    
    asyncio.create_task(publisher())
    await bus.await_event("test", timeout=5.0)
    
    assert bus.subscriber_count("test") == 0
```

**Lines covered**: 199-213 (15 lines — biggest gap!)

---

### 10. get_stats (Line 217)

```python
# Line 217: get_stats
"total_subscribers": sum(
    len(self._subscribers.get(t, [])) + len(self._async_subscribers.get(t, []))
    for t in self.topics
),
```

**Test Needed**:
```python
def test_get_stats():
    bus = EventBus()
    bus.subscribe("a", lambda e: None)
    bus.subscribe("b", lambda e: None)
    bus.subscribe_async("a", lambda e: None)
    bus.publish(Event(topic="c", payload=1))
    
    stats = bus.get_stats()
    assert stats["total_subscribers"] == 3
    assert stats["history_size"] == 1
    assert stats["max_history"] == 10000
    assert "a" in stats["topics"]
    assert "b" in stats["topics"]
```

**Lines covered**: 217 (1 line)

---

## Summary

| Gap Category | Lines | Tests Needed | Coverage Gain |
|---|---|---|---|
| Input validation | 85,87,89,98-106 | 7 tests | +11 lines |
| Unsubscribe async | 113 | 1 test | +1 line |
| History overflow | 125 | 1 test | +1 line |
| Exception handling | 131-132,136-139 | 2 tests | +4 lines |
| Async callback | 145-148 | 2 tests | +4 lines |
| publish_async | 152 | 1 test | +1 line |
| clear_history filter | 163-166 | 2 tests | +4 lines |
| subscriber_count | 175-177 | 1 test | +3 lines |
| **await_event** | **199-213** | **5 tests** | **+15 lines** |
| get_stats | 217 | 1 test | +1 line |
| **Total** | **44 lines** | **23 tests** | **+44 lines** |

## Expected Result

After implementing all 23 tests:
- **Before**: 59% (63/107 lines)
- **After**: ~95-100% (102-107/107 lines)

## Priority Order

1. **await_event** (15 lines) — largest single gap, critical for async workflows
2. **Input validation** (11 lines) — security and robustness
3. **Async callbacks** (4 lines) — async execution path
4. **Exception handling** (4 lines) — fault tolerance
5. **clear_history filter** (4 lines) — data management
6. **subscriber_count** (3 lines) — monitoring
7. **Remaining** (3 lines) — easy wins
