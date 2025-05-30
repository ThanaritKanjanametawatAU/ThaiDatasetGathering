"""Batch processing monitoring for audio enhancement."""
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BatchProcessingMonitor:
    """Monitors batch processing performance and progress."""

    def __init__(self, batch_size: int = 32):
        """Initialize batch processing monitor.

        Args:
            batch_size: Size of processing batches
        """
        self.batch_size = batch_size

        # Batch tracking
        self.batches: Dict[int, Dict] = {}
        self.current_batch_id = 0
        self.batch_start_times: Dict[int, float] = {}

        # Sample tracking
        self.samples: Dict[str, Dict] = {}
        self.sample_order: List[str] = []

        # Performance metrics
        self.processing_times: List[float] = []
        self.batch_times: List[float] = []

        # Failure tracking
        self.failures: List[Dict] = []

    def start_batch(self, batch_id: Optional[int] = None) -> int:
        """Start tracking a new batch.

        Args:
            batch_id: Optional batch ID (auto-generated if not provided)

        Returns:
            Batch ID
        """
        if batch_id is None:
            batch_id = self.current_batch_id
            self.current_batch_id += 1

        self.batch_start_times[batch_id] = time.time()
        self.batches[batch_id] = {
            'id': batch_id,
            'start_time': datetime.now().isoformat(),
            'samples': [],
            'completed': False,
            'success_count': 0,
            'failure_count': 0
        }

        return batch_id

    def record_sample(self, sample_id: str,
                      processing_time: Optional[float] = None,
                      success: bool = True,
                      error_message: Optional[str] = None,
                      batch_id: Optional[int] = None):
        """Record processing of a single sample.

        Args:
            sample_id: Sample identifier
            processing_time: Time taken to process (seconds)
            success: Whether processing succeeded
            error_message: Error message if failed
            batch_id: Batch this sample belongs to
        """
        # Determine batch
        if batch_id is None:
            # Find the latest incomplete batch
            for bid in sorted(self.batches.keys(), reverse=True):
                if not self.batches[bid]['completed']:
                    batch_id = bid
                    break

        # Record sample
        sample_data = {
            'id': sample_id,
            'timestamp': datetime.now().isoformat(),
            'processing_time': processing_time,
            'success': success,
            'error': error_message,
            'batch_id': batch_id
        }

        self.samples[sample_id] = sample_data
        self.sample_order.append(sample_id)

        # Update batch
        if batch_id is not None and batch_id in self.batches:
            self.batches[batch_id]['samples'].append(sample_id)
            if success:
                self.batches[batch_id]['success_count'] += 1
            else:
                self.batches[batch_id]['failure_count'] += 1

        # Track processing time
        if processing_time is not None:
            self.processing_times.append(processing_time)

        # Track failures
        if not success:
            self.failures.append({
                'sample_id': sample_id,
                'timestamp': datetime.now().isoformat(),
                'error': error_message or 'Unknown error',
                'batch_id': batch_id
            })

    def complete_batch(self, batch_id: int):
        """Mark a batch as completed.

        Args:
            batch_id: Batch ID to complete
        """
        if batch_id not in self.batches:
            logger.warning(f"Batch {batch_id} not found")
            return

        batch = self.batches[batch_id]
        batch['completed'] = True
        batch['end_time'] = datetime.now().isoformat()

        # Calculate batch processing time
        if batch_id in self.batch_start_times:
            batch_time = time.time() - self.batch_start_times[batch_id]
            batch['processing_time'] = batch_time
            self.batch_times.append(batch_time)

    def get_statistics(self) -> Dict:
        """Get overall processing statistics.

        Returns:
            Dictionary of statistics
        """
        total_samples = len(self.samples)
        successful_samples = sum(1 for s in self.samples.values() if s['success'])

        stats = {
            'total_batches': len(self.batches),
            'completed_batches': sum(1 for b in self.batches.values() if b['completed']),
            'total_samples': total_samples,
            'successful_samples': successful_samples,
            'failed_samples': len(self.failures),
            'success_rate': (successful_samples / total_samples * 100) if total_samples > 0 else 0
        }

        # Processing time statistics
        if self.processing_times:
            stats['avg_sample_time'] = np.mean(self.processing_times)
            stats['min_sample_time'] = np.min(self.processing_times)
            stats['max_sample_time'] = np.max(self.processing_times)
            stats['std_sample_time'] = np.std(self.processing_times)
        else:
            stats['avg_sample_time'] = 0
            stats['min_sample_time'] = 0
            stats['max_sample_time'] = 0
            stats['std_sample_time'] = 0

        # Batch time statistics
        if self.batch_times:
            stats['avg_batch_time'] = np.mean(self.batch_times)
            stats['throughput_per_minute'] = 60 / stats['avg_sample_time'] if stats['avg_sample_time'] > 0 else 0
        else:
            stats['avg_batch_time'] = 0
            stats['throughput_per_minute'] = 0

        return stats

    def get_failures(self) -> List[Dict]:
        """Get list of all failures.

        Returns:
            List of failure records
        """
        return self.failures.copy()

    def detect_bottlenecks(self, threshold_factor: float = 3.0) -> List[Dict]:
        """Detect samples that took unusually long to process.

        Args:
            threshold_factor: Factor of average time to consider bottleneck

        Returns:
            List of bottleneck samples
        """
        if not self.processing_times:
            return []

        avg_time = np.mean(self.processing_times)
        threshold = avg_time * threshold_factor

        bottlenecks = []
        for sample_id, sample in self.samples.items():
            if sample.get('processing_time', 0) > threshold:
                bottlenecks.append({
                    'sample_id': sample_id,
                    'processing_time': sample['processing_time'],
                    'factor': sample['processing_time'] / avg_time,
                    'batch_id': sample.get('batch_id')
                })

        # Sort by processing time
        bottlenecks.sort(key=lambda x: x['processing_time'], reverse=True)

        return bottlenecks

    def get_batch_summary(self, batch_id: int) -> Optional[Dict]:
        """Get summary for a specific batch.

        Args:
            batch_id: Batch ID

        Returns:
            Batch summary or None if not found
        """
        if batch_id not in self.batches:
            return None

        batch = self.batches[batch_id]

        # Get sample times for this batch
        sample_times = []
        for sample_id in batch['samples']:
            if sample_id in self.samples:
                sample_time = self.samples[sample_id].get('processing_time')
                if sample_time is not None:
                    sample_times.append(sample_time)

        summary = {
            'batch_id': batch_id,
            'sample_count': len(batch['samples']),
            'success_count': batch['success_count'],
            'failure_count': batch['failure_count'],
            'success_rate': (batch['success_count'] / len(batch['samples']) * 100) if batch['samples'] else 0,
            'completed': batch['completed'],
            'total_time': batch.get('processing_time', 0),
            'avg_sample_time': np.mean(sample_times) if sample_times else 0,
            'start_time': batch['start_time'],
            'end_time': batch.get('end_time')
        }

        return summary

    def get_recent_performance(self, sample_count: int = 100) -> Dict:
        """Get performance metrics for recent samples.

        Args:
            sample_count: Number of recent samples to analyze

        Returns:
            Performance metrics
        """
        # Get recent samples
        recent_sample_ids = self.sample_order[-sample_count:]
        recent_samples = [self.samples[sid] for sid in recent_sample_ids if sid in self.samples]

        if not recent_samples:
            return {
                'sample_count': 0,
                'avg_time': 0,
                'success_rate': 0,
                'trend': 'stable'
            }

        # Calculate metrics
        times = [s['processing_time'] for s in recent_samples if s.get('processing_time') is not None]
        successes = sum(1 for s in recent_samples if s['success'])

        # Calculate trend
        if len(times) >= 10:
            first_half = times[:len(times)//2]
            second_half = times[len(times)//2:]

            if np.mean(second_half) > np.mean(first_half) * 1.2:
                trend = 'slowing'
            elif np.mean(second_half) < np.mean(first_half) * 0.8:
                trend = 'improving'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'

        return {
            'sample_count': len(recent_samples),
            'avg_time': np.mean(times) if times else 0,
            'success_rate': (successes / len(recent_samples) * 100) if recent_samples else 0,
            'trend': trend
        }

    def estimate_completion_time(self, remaining_samples: int) -> Tuple[float, str]:
        """Estimate time to complete remaining samples.

        Args:
            remaining_samples: Number of samples left to process

        Returns:
            Tuple of (seconds, formatted_time_string)
        """
        stats = self.get_statistics()

        if stats['avg_sample_time'] == 0:
            return 0, "Unknown"

        # Estimate based on recent performance
        recent_perf = self.get_recent_performance()
        avg_time = recent_perf['avg_time'] if recent_perf['avg_time'] > 0 else stats['avg_sample_time']

        # Add overhead for batch processing
        overhead_factor = 1.1  # 10% overhead

        estimated_seconds = remaining_samples * avg_time * overhead_factor

        # Format time
        if estimated_seconds < 60:
            formatted = f"{int(estimated_seconds)}s"
        elif estimated_seconds < 3600:
            minutes = int(estimated_seconds / 60)
            seconds = int(estimated_seconds % 60)
            formatted = f"{minutes}m {seconds}s"
        else:
            hours = int(estimated_seconds / 3600)
            minutes = int((estimated_seconds % 3600) / 60)
            formatted = f"{hours}h {minutes}m"

        return estimated_seconds, formatted

    def get_efficiency_report(self) -> Dict:
        """Generate efficiency report.

        Returns:
            Efficiency metrics and recommendations
        """
        stats = self.get_statistics()

        # Calculate efficiency metrics
        if self.batch_times and len(self.batches) > 0:
            avg_batch_samples = sum(len(b['samples']) for b in self.batches.values()) / len(self.batches)
            ideal_batch_time = avg_batch_samples * stats['avg_sample_time']
            actual_batch_time = stats['avg_batch_time']

            efficiency = (ideal_batch_time / actual_batch_time * 100) if actual_batch_time > 0 else 0
        else:
            efficiency = 0
            avg_batch_samples = 0

        # Identify issues
        issues = []
        recommendations = []

        if stats['success_rate'] < 95:
            issues.append(f"High failure rate: {100 - stats['success_rate']:.1f}%")
            recommendations.append("Investigate common failure patterns")

        if stats['std_sample_time'] > stats['avg_sample_time'] * 0.5:
            issues.append("High processing time variance")
            recommendations.append("Identify and optimize bottleneck samples")

        if efficiency < 80:
            issues.append(f"Low batch efficiency: {efficiency:.1f}%")
            recommendations.append("Consider adjusting batch size or processing strategy")

        bottlenecks = self.detect_bottlenecks()
        if len(bottlenecks) > len(self.samples) * 0.05:  # More than 5% are bottlenecks
            issues.append(f"{len(bottlenecks)} samples taking unusually long")
            recommendations.append("Profile slow samples to identify optimization opportunities")

        return {
            'efficiency_percentage': efficiency,
            'avg_batch_utilization': avg_batch_samples / self.batch_size * 100 if self.batch_size > 0 else 0,
            'issues': issues,
            'recommendations': recommendations,
            'bottleneck_count': len(bottlenecks),
            'failure_rate': 100 - stats['success_rate']
        }