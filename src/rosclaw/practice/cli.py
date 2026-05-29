"""Practice CLI - Episode recording and replay."""
import json
from pathlib import Path


def list_episodes():
    artifacts = Path('.rosclaw/artifacts/episodes')
    if not artifacts.exists():
        print('No episodes recorded yet.')
        return
    for d in sorted(artifacts.glob('ep_*')):
        meta = d / 'metadata.json'
        if meta.exists():
            data = json.load(open(meta))
            print(f"{d.name}: {data.get('task_id', 'unknown')}")
        else:
            print(d.name)


def show_episode(episode_id):
    meta_path = Path(f'.rosclaw/artifacts/episodes/{episode_id}/metadata.json')
    if not meta_path.exists():
        print(f'Episode {episode_id} not found.')
        return
    data = json.load(open(meta_path))
    print(json.dumps(data, indent=2))


def replay_episode(episode_id):
    events_path = Path(f'.rosclaw/artifacts/episodes/{episode_id}/events.jsonl')
    if not events_path.exists():
        print(f'Episode {episode_id} has no events.')
        return
    for line in open(events_path):
        event = json.loads(line)
        print(f"[{event.get('timestamp', '?')}] {event.get('type', '?')}")
