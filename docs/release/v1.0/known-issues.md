# ROSClaw v1.0 Known Issues

## Pre-existing Issues (Not from current integration sprint)

### SeekDBSQLiteClient Missing Abstract Methods
- **File**: `src/rosclaw/memory/seekdb_client.py`
- **Issue**: `SeekDBSQLiteClient` is missing `delete` and `delete_where` method implementations
- **Impact**: `test_seekdb_indexes.py` fails with `TypeError: Can't instantiate abstract class SeekDBSQLiteClient without an implementation for abstract methods 'delete', 'delete_where'`
- **Status**: Pre-existing, not introduced by current integration changes
- **Plan**: Fix in v1.1

## Integration Notes

See [INTEGRATION_STATUS.md](INTEGRATION_STATUS.md) for current integration progress.
