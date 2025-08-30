import argparse
import sys
import os
import time
import subprocess
from pathlib import Path
from typing import Dict, Optional

def main():
    """Production CLI with comprehensive features"""
    parser = argparse.ArgumentParser(description='Production Webinar Video Reframer')
    
    parser.add_argument('--input', '-i', required=True, help='Input webinar video')
    parser.add_argument('--output', '-o', required=True, help='Output reframed video')
    parser.add_argument('--quality', choices=['fast', 'balanced', 'high'], default='balanced',
                       help='Processing quality preset')
    parser.add_argument('--bitrate', default='8M', help='Target bitrate (e.g., 8M, 6M)')
    parser.add_argument('--preview', action='store_true', help='Generate preview (first 30s)')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    
    args = parser.parse_args()
    
    # Import here to avoid import issues
    sys.path.append('.')
    from src.production_pipeline import ProductionPipeline
    from src.edge_cases.handler import EdgeCaseHandler
    
    # Validate input
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure based on quality preset
    config = _get_quality_config(args.quality)
    config['target_bitrate'] = args.bitrate
    
    print(f"=== Production Webinar Reframer ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Quality: {args.quality}")
    print(f"Bitrate: {args.bitrate}")
    
    if args.preview:
        print("Mode: Preview (30 seconds)")
        preview_input = _create_preview_clip(args.input)
        if preview_input:
            args.input = preview_input
    
    # Initialize pipeline
    pipeline = ProductionPipeline(config)
    edge_handler = EdgeCaseHandler()
    
    # Progress callback
    def show_progress(percent, current, total):
        print(f"\rProgress: {percent:6.1f}% [{current:5d}/{total:5d}] ", end="", flush=True)
    
    # Process video
    start_time = time.time()
    
    try:
        success = pipeline.process_video_production(
            args.input, args.output, show_progress
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n\nProcessing completed in {processing_time:.1f} seconds")
        
        if success:
            print(f"✓ Success! Output saved to: {args.output}")
            
            # Show edge case statistics
            failure_report = edge_handler.get_failure_report()
            if any(failure_report.values()):
                print("\nEdge cases handled:")
                for case, count in failure_report.items():
                    if count > 0:
                        print(f"  {case}: {count} instances")
        else:
            print("✗ Processing failed")
            return 1
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0

def _get_quality_config(quality_preset: str) -> Dict:
    """Get configuration for quality preset"""
    
    if quality_preset == 'fast':
        return {
            'min_face_conf': 0.4,
            'shot_threshold': 0.4,
            'face_alpha': 0.5,
            'slide_window': 5,
            'preprocess_input': False,
            'enable_quality_enhancement': False
        }
    elif quality_preset == 'high':
            return {
                'min_face_conf': 0.2,
                'shot_threshold': 0.2,
                'face_alpha': 0.2,
                'slide_window': 12,
                'preprocess_input': True,
                'enable_quality_enhancement': True,
                'enable_stability_analysis': True
            }
    else:  # balanced
            return {
                'min_face_conf': 0.3,
                'shot_threshold': 0.3,
                'face_alpha': 0.3,
                'slide_window': 8,
                'preprocess_input': True,
                'enable_quality_enhancement': True
            }

def _create_preview_clip(input_video: str) -> Optional[str]:
    """Create 30-second preview clip for testing"""
    try:
        preview_path = "temp_preview.mp4"
        
        cmd = [
            'ffmpeg', '-y',
            '-i', input_video,
            '-t', '30',  # 30 seconds
            '-c', 'copy',
            preview_path
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode == 0 and os.path.exists(preview_path):
            print(f"Preview clip created: {preview_path}")
            return preview_path
        
        return None
        
    except Exception as e:
        print(f"Preview creation failed: {e}")
        return None

if __name__ == "__main__":
    sys.exit(main())