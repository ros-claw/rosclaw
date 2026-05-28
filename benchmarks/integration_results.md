# ROSClaw v1.0 Integration Performance Benchmark

> **Date**: 2026-05-29
> **Method**: In-process synthetic benchmark

## KNOW Query Latency
- Queries: 1,000
- p50: 0.0002 ms
- p95: 0.0002 ms
- p99: 0.0003 ms
- Range: 0.0002 - 0.0145 ms
- Target: < 100.0000 ms
- Verdict: PASS

## HOW Recovery Latency
- Queries: 1,000
- p50: 0.0013 ms
- p95: 0.0013 ms
- p99: 0.0015 ms
- Range: 0.0013 - 0.6046 ms
- Target: < 10.0000 ms
- Verdict: PASS

## EventBus Throughput (Multi-Subscriber)
- Events: 10,000
- Elapsed: 0.0876s
- Throughput: 114214.9 events/s
- Received: 30,000 (5 subscribers)
- Target: >= 10,000 events/s
- Verdict: PASS

## SeekDB Query Latency
- Queries: 1,000
- p50: 3.5902 ms
- p95: 6.4632 ms
- p99: 9.5842 ms
- Target: < 50.0000 ms
- Verdict: PASS

## End-to-End Pipeline Latency
- Iterations: 100
- p50: 0.0041 ms
- p95: 0.0049 ms
- p99: 0.0171 ms
- Target: < 500.0000 ms
- Verdict: PASS
