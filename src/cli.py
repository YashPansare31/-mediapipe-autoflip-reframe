import argparse
import sys
import os
import time
from pathlib import Path
from typing import Dict, Optional

def create_parser() -> argparse.ArgumentParser:
    """Create comprehensive CLI parser"""
    
    parser = argparse.ArgumentParser(
        prog='webinar-reframer',
        description='Automatically reframe horizontal webinar videos to vertical 9:16 format',
        epilog='Examples:\n'
               '  %(prog)s -i webinar.mp4 -o reframed.mp4\n'
               '  %(prog)s -i input.mp4 -o output.mp4 --quality high --preview\n'
               '  %(prog)s -i video.mp4 -o result.mp4 --bitrate 6M --no-autoflip',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('-i', '--input', required=True, metavar='FILE',
                       help='Input webinar video file')
    parser.add_argument('-o', '--output', required=True, metavar='FILE',
                       help='Output reframed video file')
    
    # Quality settings
    quality_group = parser.add_argument_group('Quality Settings')
    quality_group.add_argument('--quality', choices=['fast', 'balanced', 'high'], 
                              default='balanced',
                              help='Processing quality preset (default: balanced)')
    quality_group.add_argument('--bitrate', default='8M', metavar='RATE',
                              help='Target bitrate, e.g. 8M, 6M, 4M (default: 8M)')
    
    # Detection settings
    detection_group = parser.add_argument_group('Detection Settings')
    detection_group.add_argument('--min-face-conf', type=float, default=0.3, metavar='CONF',
                                help='Minimum face detection confidence 0.0-1.0 (default: 0.3)')
    detection_group.add_argument('--face-tracking', type=float, default=0.3, metavar='ALPHA',
                                help='Face tracking smoothing 0.1-0.7 (default: 0.3)')
    
    # Processing options
    process_group = parser.add_argument_group('Processing Options')
    process_group.add_argument('--no-autoflip', action='store_true',
                              help='Use manual composition instead of AutoFlip')
    process_group.add_argument('--preview', action='store_true',
                              help='Process only first 30 seconds for testing')
    process_group.add_argument('--preprocess', action='store_true', 
                              help='Preprocess input video for better compatibility')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--benchmark', action='store_true',
                             help='Show detailed performance metrics')
    output_group.add_argument('--quiet', '-q', action='store_true',
                             help='Suppress progress output')
    output_group.add_argument('--verbose', '-v', action='store_true',
                             help='Show detailed processing information')
    
    return parser

def validate_args(args) -> bool:
    """Validate command line arguments"""
    
    # Check input file
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return False
    
    # Check input is video file
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
    if Path(args.input).suffix.lower() not in video_extensions:
        print(f"Warning: Input file may not be a video: {args.input}")
    
    # Validate confidence range
    if not 0.0 <= args.min_face_conf <= 1.0:
        print(f"Error: Face confidence must be between 0.0 and 1.0")
        return False
    
    # Validate tracking range
    if not 0.1 <= args.face_tracking <= 0.7:
        print(f"Error: Face tracking must be between 0.1 and 0.7")
        return False
    
    # Create output directory
    output_dir = Path(args.output).parent
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Error: Cannot create output directory: {e}")
            return False
    
    return True

def get_config_from_args(args) -> Dict:
    """Convert CLI args to pipeline configuration"""
    
    quality_configs = {
        'fast': {
            'min_face_conf': args.min_face_conf,
            'shot_threshold': 0.4,
            'face_alpha': 0.5,
            'slide_window': 5,
            'preprocess_input': False,
            'enable_quality_enhancement': False
        },
        'balanced': {
            'min_face_conf': args.min_face_conf,
            'shot_threshold': 0.3,
            'face_alpha': args.face_tracking,
            'slide_window': 8,
            'preprocess_input': True,
            'enable_quality_enhancement': True
        },
        'high': {
            'min_face_conf': max(0.2, args.min_face_conf - 0.1),
            'shot_threshold': 0.2,
            'face_alpha': args.face_tracking * 0.7,
            'slide_window': 12,
            'preprocess_input': True,
            'enable_quality_enhancement': True,
            'enable_stability_analysis': True
        }
    }
    
    config = quality_configs[args.quality].copy()
    config['target_bitrate'] = args.bitrate
    
    if args.preprocess:
        config['preprocess_input'] = True
    
    return config

def main() -> int:
    """Main CLI entry point"""
    
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not validate_args(args):
        return 1
    
    # Show banner
    if not args.quiet:
        print("=" * 50)
        print("    Webinar Video Reframer v1.0")
        print("    9:16 Vertical Format Converter")
        print("=" * 50)
    
    # Import here to avoid slow startup for --help
    sys.path.append('.')
    try:
        from src.production_pipeline import ProductionPipeline
    except ImportError as e:
        print(f"Error: Cannot import pipeline modules: {e}")
        print("Make sure you're running from the project root directory")
        return 1
    
    # Get configuration
    config = get_config_from_args(args)
    
    if args.verbose:
        print(f"Configuration: {config}")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Quality: {args.quality}")
        print(f"AutoFlip: {'Disabled' if args.no_autoflip else 'Enabled'}")
    
    # Progress callback
    def progress_callback(percent, current, total):
        if not args.quiet:
            print(f"\rProgress: {percent:6.1f}% [{current:5d}/{total:5d}] ", 
                  end="", flush=True)
    
    # Initialize pipeline
    try:
        pipeline = ProductionPipeline(config)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return 1
    
    # Process video
    start_time = time.time()
    
    try:
        success = pipeline.process_video_production(
            args.input, 
            args.output, 
            progress_callback if not args.quiet else None
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if not args.quiet:
            print(f"\n")  # New line after progress
        
        if success:
            if not args.quiet:
                print(f"✓ Success! Video reframed in {processing_time:.1f}s")
                print(f"✓ Output saved to: {args.output}")
                
                # Show file info
                if os.path.exists(args.output):
                    output_size = os.path.getsize(args.output) / (1024 * 1024)
                    print(f"✓ File size: {output_size:.1f} MB")
            
            if args.benchmark:
                # Show performance metrics
                metrics = pipeline.performance_optimizer.metrics
                print(f"\nPerformance Metrics:")
                print(f"  Processing FPS: {metrics.fps_processing:.1f}")
                print(f"  Memory used: {metrics.memory_usage_mb:.1f} MB")
                print(f"  CPU usage: {metrics.cpu_usage_percent:.1f}%")
            
            return 0
        else:
            print("✗ Processing failed")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n✗ Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Processing error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())