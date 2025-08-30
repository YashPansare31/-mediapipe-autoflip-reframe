import argparse
import sys
from pathlib import Path
from .pipeline import WebinarReframingPipeline

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Webinar Video Reframer')
    
    parser.add_argument('--input', '-i', required=True, 
                       help='Input webinar video file')
    parser.add_argument('--output', '-o', required=True,
                       help='Output reframed video file')
    parser.add_argument('--config', '-c', 
                       default='configs/portrait_916.pbtxt',
                       help='AutoFlip configuration file')
    parser.add_argument('--min-face-conf', type=float, default=0.3,
                       help='Minimum face detection confidence')
    parser.add_argument('--stabilize', action='store_true',
                       help='Enable temporal stabilization')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output and visualization')
    parser.add_argument('--no-autoflip', action='store_true',
                       help='Use manual composition instead of AutoFlip')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found")
        return 1
    
    print(f"Processing: {args.input}")
    print(f"Output: {args.output}")
    print(f"Face confidence threshold: {args.min_face_conf}")
    print(f"Stabilization: {'ON' if args.stabilize else 'OFF'}")
    print(f"AutoFlip: {'OFF' if args.no_autoflip else 'ON'}")
    
    # Create pipeline configuration
    config = {
        'min_face_conf': args.min_face_conf,
        'autoflip_config': args.config if Path(args.config).exists() else None
    }
    
    # Initialize and run pipeline
    pipeline = WebinarReframingPipeline(config)
    
    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Process video
    success = pipeline.process_video(
        args.input, 
        args.output, 
        use_autoflip=not args.no_autoflip
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())