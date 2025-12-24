"""
Batch Calculator Utility for GodsEye Pipeline

Calculates optimal batch sizes for parallel processing across pipeline stages.
"""

import math
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class BatchInfo:
    """Information about a single batch"""
    batch_id: int
    start_idx: int
    end_idx: int
    size: int
    items: List[Any] = None  # Optional: actual items in this batch
    
    def __post_init__(self):
        if self.items is None:
            self.items = []


def calculate_batches(total_items: int, max_parallel: int) -> List[BatchInfo]:
    """
    Calculate number of batches based on items and parallel limit.
    
    Args:
        total_items: Total number of items to process
        max_parallel: Maximum number of parallel workers per batch
        
    Returns:
        List of BatchInfo objects with batch details
        
    Example:
        >>> batches = calculate_batches(50, 10)
        >>> len(batches)  # 5 batches
        5
        >>> batches[0].size  # First batch has 10 items
        10
        >>> batches[-1].size  # Last batch has 10 items
        10
    """
    if total_items <= 0:
        return []
    
    if max_parallel <= 0:
        raise ValueError("max_parallel must be greater than 0")
    
    num_batches = math.ceil(total_items / max_parallel)
    batches = []
    
    for i in range(num_batches):
        start_idx = i * max_parallel
        end_idx = min((i + 1) * max_parallel, total_items)
        batch_size = end_idx - start_idx
        
        batch = BatchInfo(
            batch_id=i + 1,
            start_idx=start_idx,
            end_idx=end_idx,
            size=batch_size
        )
        batches.append(batch)
    
    return batches


def create_batches_with_items(items: List[Any], max_parallel: int) -> List[BatchInfo]:
    """
    Create batches with actual items assigned to each batch.
    
    Args:
        items: List of items to batch
        max_parallel: Maximum number of parallel workers per batch
        
    Returns:
        List of BatchInfo objects with items populated
    """
    total_items = len(items)
    batch_infos = calculate_batches(total_items, max_parallel)
    
    # Assign actual items to batches
    for batch in batch_infos:
        start, end = batch.start_idx, batch.end_idx
        batch.items = items[start:end]
    
    return batch_infos


def get_batch_summary(batches: List[BatchInfo]) -> Dict[str, Any]:
    """
    Get summary statistics for a list of batches.
    
    Args:
        batches: List of BatchInfo objects
        
    Returns:
        Dictionary with batch summary statistics
    """
    if not batches:
        return {
            'total_batches': 0,
            'total_items': 0,
            'avg_batch_size': 0,
            'max_batch_size': 0,
            'min_batch_size': 0
        }
    
    total_items = sum(batch.size for batch in batches)
    batch_sizes = [batch.size for batch in batches]
    
    return {
        'total_batches': len(batches),
        'total_items': total_items,
        'avg_batch_size': total_items / len(batches),
        'max_batch_size': max(batch_sizes),
        'min_batch_size': min(batch_sizes),
        'batch_details': [
            {
                'batch_id': batch.batch_id,
                'size': batch.size,
                'start_idx': batch.start_idx,
                'end_idx': batch.end_idx
            }
            for batch in batches
        ]
    }


def validate_batch_configuration(total_items: int, max_parallel: int) -> Dict[str, Any]:
    """
    Validate batch configuration and return recommendations.
    
    Args:
        total_items: Total number of items to process
        max_parallel: Maximum number of parallel workers per batch
        
    Returns:
        Dictionary with validation results and recommendations
    """
    issues = []
    recommendations = []
    
    # Check for edge cases
    if total_items == 0:
        issues.append("No items to process")
        return {'valid': False, 'issues': issues, 'recommendations': recommendations}
    
    if max_parallel <= 0:
        issues.append("max_parallel must be greater than 0")
        return {'valid': False, 'issues': issues, 'recommendations': recommendations}
    
    # Performance recommendations
    if max_parallel > 50:
        recommendations.append("Consider reducing max_parallel to avoid resource exhaustion")
    
    if max_parallel < 2:
        recommendations.append("Consider increasing max_parallel for better parallelism")
    
    # Calculate efficiency
    batches = calculate_batches(total_items, max_parallel)
    last_batch_size = batches[-1].size if batches else 0
    
    if last_batch_size < max_parallel * 0.5 and len(batches) > 1:
        recommendations.append(f"Last batch is only {last_batch_size} items, consider adjusting max_parallel")
    
    # Calculate total parallel slots used
    total_slots = len(batches) * max_parallel
    utilization = total_items / total_slots if total_slots > 0 else 0
    
    if utilization < 0.7:
        recommendations.append(f"Worker utilization is {utilization:.1%}, consider reducing max_parallel")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'recommendations': recommendations,
        'utilization': utilization,
        'batch_count': len(batches),
        'last_batch_size': last_batch_size
    }


if __name__ == "__main__":
    # Example usage and testing
    print("=== Batch Calculator Test ===")
    
    # Test case 1: Perfect division
    print("\nTest 1: 50 items, max_parallel=10")
    batches1 = calculate_batches(50, 10)
    summary1 = get_batch_summary(batches1)
    print(f"Batches: {summary1['total_batches']}, Items per batch: {[b.size for b in batches1]}")
    
    # Test case 2: Uneven division
    print("\nTest 2: 35 items, max_parallel=10")
    batches2 = calculate_batches(35, 10)
    summary2 = get_batch_summary(batches2)
    print(f"Batches: {summary2['total_batches']}, Items per batch: {[b.size for b in batches2]}")
    
    # Test case 3: Single batch
    print("\nTest 3: 5 items, max_parallel=10")
    batches3 = calculate_batches(5, 10)
    summary3 = get_batch_summary(batches3)
    print(f"Batches: {summary3['total_batches']}, Items per batch: {[b.size for b in batches3]}")
    
    # Test validation
    print("\nTest 4: Validation for 50 items, max_parallel=10")
    validation = validate_batch_configuration(50, 10)
    print(f"Valid: {validation['valid']}")
    print(f"Utilization: {validation['utilization']:.1%}")
    if validation['recommendations']:
        print("Recommendations:", validation['recommendations'])
