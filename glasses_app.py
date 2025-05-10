import os
import sys
import argparse
import threading
import time
import webbrowser
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description='Hunyuan3D Glasses Application')
    parser.add_argument('--mode', type=str, choices=['setup', 'generate', 'customize', 'tryon', 'all'], 
                        default='all', help='Mode to run')
    parser.add_argument('--data_dir', type=str, default='data/glasses', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None, help='Path to input image for generation')
    parser.add_argument('--glasses_model', type=str, default=None, help='Path to glasses model for customization')
    parser.add_argument('--face_image', type=str, default=None, help='Path to face image for try-on')
    parser.add_argument('--port_generate', type=int, default=7860, help='Port for generation web interface')
    parser.add_argument('--port_customize', type=int, default=7861, help='Port for customization web interface')
    parser.add_argument('--download_hunyuan', action='store_true', help='Download Hunyuan3D-2 model')
    parser.add_argument('--no_browser', action='store_true', help='Do not open browser automatically')
    return parser.parse_args()

def run_setup(args):
    """Run the setup script"""
    print("Setting up the environment...")
    
    cmd = [
        sys.executable, 'setup_glasses.py',
        '--data_dir', args.data_dir
    ]
    
    if args.download_hunyuan:
        cmd.append('--download_hunyuan')
    
    cmd.extend(['--create_dirs', '--install_deps'])
    
    subprocess.run(cmd)
    print("Setup completed!")

def run_generate(args):
    """Run the generation script or web interface"""
    print("Starting glasses generation...")
    
    if args.image:
        # Run the demo script with the provided image
        cmd = [
            sys.executable, 'src/glasses_demo.py',
            '--image', args.image,
            '--output_dir', args.output_dir
        ]
        
        if args.checkpoint:
            cmd.extend(['--checkpoint', args.checkpoint])
        
        subprocess.run(cmd)
        print(f"Generation completed! Results saved to {args.output_dir}")
    else:
        # Start the web interface
        print(f"Starting web interface on port {args.port_generate}...")
        
        cmd = [
            sys.executable, 'src/web_interface.py',
            '--output_dir', args.output_dir,
            '--port', str(args.port_generate)
        ]
        
        if args.checkpoint:
            cmd.extend(['--checkpoint', args.checkpoint])
        
        # Start in a new thread
        thread = threading.Thread(target=lambda: subprocess.run(cmd))
        thread.daemon = True
        thread.start()
        
        # Open browser
        if not args.no_browser:
            time.sleep(3)  # Wait for server to start
            webbrowser.open(f"http://localhost:{args.port_generate}")
        
        return thread

def run_customize(args):
    """Run the customization web interface"""
    print("Starting glasses customization interface...")
    
    cmd = [
        sys.executable, 'src/glasses_customizer.py',
        '--output_dir', args.output_dir,
        '--port', str(args.port_customize)
    ]
    
    # Start in a new thread
    thread = threading.Thread(target=lambda: subprocess.run(cmd))
    thread.daemon = True
    thread.start()
    
    # Open browser
    if not args.no_browser:
        time.sleep(3)  # Wait for server to start
        webbrowser.open(f"http://localhost:{args.port_customize}")
    
    return thread

def run_tryon(args):
    """Run the virtual try-on script"""
    print("Running virtual try-on...")
    
    if not args.glasses_model or not args.face_image:
        print("Error: Both --glasses_model and --face_image are required for try-on mode")
        return
    
    cmd = [
        sys.executable, 'src/virtual_tryon.py',
        '--glasses', args.glasses_model,
        '--face', args.face_image,
        '--output', os.path.join(args.output_dir, 'tryon_result.jpg')
    ]
    
    subprocess.run(cmd)
    print(f"Try-on completed! Result saved to {os.path.join(args.output_dir, 'tryon_result.jpg')}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the selected mode
    if args.mode == 'setup' or args.mode == 'all':
        run_setup(args)
    
    threads = []
    
    if args.mode == 'generate' or args.mode == 'all':
        thread = run_generate(args)
        if thread:
            threads.append(thread)
    
    if args.mode == 'customize' or args.mode == 'all':
        thread = run_customize(args)
        if thread:
            threads.append(thread)
    
    if args.mode == 'tryon':
        run_tryon(args)
    
    # Keep the main thread alive if we started any server threads
    if threads:
        try:
            print("\nPress Ctrl+C to exit")
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            print("\nExiting...")

if __name__ == "__main__":
    main()
