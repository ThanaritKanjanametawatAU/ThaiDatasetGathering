# Parallel Development Coordination

## Active Worktrees
Three parallel worktrees have been created for the audio quality enhancement feature:

1. **audio_quality_enhancement-core** (`./trees/audio_quality_enhancement-core`)
   - Focus: Core noise reduction engine
   - Branch: `audio_quality_enhancement-core`
   - Lead: Agent/Developer 1

2. **audio_quality_enhancement-speaker** (`./trees/audio_quality_enhancement-speaker`)
   - Focus: Secondary speaker detection
   - Branch: `audio_quality_enhancement-speaker`
   - Lead: Agent/Developer 2

3. **audio_quality_enhancement-dashboard** (`./trees/audio_quality_enhancement-dashboard`)
   - Focus: Dashboard and testing
   - Branch: `audio_quality_enhancement-dashboard`
   - Lead: Agent/Developer 3

## Merge Conflict Avoidance Strategy

### 1. Module Separation
Each worktree works on separate directories:
- Core: `processors/audio_enhancement/`, `utils/audio_quality.py`
- Speaker: `processors/speaker_detection/`, `utils/vad.py`
- Dashboard: `dashboard/`, tests, main.py integration

### 2. Shared Interfaces
Coordinate through these files (edit carefully):
- `config.py` - Configuration constants
- `processors/__init__.py` - Module exports
- `utils/__init__.py` - Utility exports

### 3. Integration Points
Use feature flags in config.py:
```python
AUDIO_ENHANCEMENT_CONFIG = {
    "enable_noise_reduction": False,  # Core team
    "enable_speaker_detection": False,  # Speaker team
    "enable_dashboard": False,  # Dashboard team
}
```

### 4. Daily Sync Process
1. Each team rebases from main: `git fetch origin && git rebase origin/master`
2. Run tests: `python -m pytest tests/test_audio_enhancement_comprehensive.py`
3. Push to branch: `git push origin <branch-name>`
4. Create PR when module is complete

## Communication Protocol
- Use PR comments for cross-team discussions
- Update this file with integration decisions
- Run comprehensive tests before merging

## Merge Order
Suggested merge sequence to minimize conflicts:
1. Core enhancement (foundation)
2. Speaker detection (builds on core)
3. Dashboard (integrates everything)

## Commands
- View all worktrees: `git worktree list`
- Switch between worktrees: `cd ./trees/<worktree-name>`
- Remove worktree: `git worktree remove <worktree-name>`