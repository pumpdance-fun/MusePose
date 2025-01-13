# ... existing imports ...
import redis
import json
from threading import Thread
import time
from video_generation import VideoGenerator
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

args = {
    "yolox_config": "./pose/config/yolox_l_8xb8-300e_coco.py",
    "yolox_ckpt": "./pretrained_weights/dwpose/yolox_l_8x8_300e_coco.pth",
    "dwpose_config": "./pose/config/dwpose-l_384x288.py",
    "dwpose_ckpt": "./pretrained_weights/dwpose/dw-ll_ucoco_384.pth",
    "align_frame": 0,
    "max_frame": 100,
    "detect_resolution": 512,
    "image_resolution": 720,
    "H": 256,
    "W": 144,
    "L": 300,
    "S": 48,
    "O": 4,
    "cfg": 3.5,
    "seed": 99,
    "steps": 20,
    "skip": 1,
    "config": "./configs/model_config.yaml",
    "num_workers": 2
}

class VideoGenerationService:
    def __init__(self, args):

        self.num_workers = args["num_workers"]
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.task_queue = "video_generation_tasks"
        self.result_queue = "video_generation_results"
        self.should_run = True
        # Create thread pool and lock
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self.lock = Lock()
        # Create a pool of video generators
        self.generators = [VideoGenerator(args) for _ in range(self.num_workers)]
        self.generator_index = 0


    def _get_next_generator(self):
        """Get next available generator from the pool using round-robin"""
        with self.lock:
            generator = self.generators[self.generator_index]
            self.generator_index = (self.generator_index + 1) % self.num_workers
            return generator
        
    def start(self):
        """Start the service in a background thread"""
        self.thread = Thread(target=self._process_queue)
        self.thread.daemon = True
        self.thread.start()
        print("Video Generation Service started")

    def stop(self):
        """Stop the service"""
        self.should_run = False
        self.thread.join()
        self.executor.shutdown(wait=True)
        print("Video Generation Service stopped")


    def _process_task(self, task_data):
        """Process a single task with an available generator"""
        try:
            task_dict = json.loads(task_data)
            print(f"Processing task: {task_dict['task_id']}")
            
            # Get an available generator from the pool
            generator = self._get_next_generator()
            
            # Process the video generation
            output_path = generator(
                task_dict['ref_image_path'],
                task_dict['ref_video_path'],
                task_dict['save_dir']
            )
            
            result = {
                'task_id': task_dict['task_id'],
                'status': 'completed',
                'output_path': output_path
            }
            
        except Exception as e:
            result = {
                'task_id': task_dict['task_id'],
                'status': 'failed',
                'error': str(e)
            }
        
        # Push result to result queue
        self.redis_client.rpush(
            self.result_queue,
            json.dumps(result)
        )

    def _process_queue(self):
        """Main loop to process tasks from the queue"""
        while self.should_run:
            # Get task from Redis queue with timeout
            task = self.redis_client.blpop(self.task_queue, timeout=1)
            
            if task:
                _, task_data = task
                # Submit task to thread pool
                self.executor.submit(self._process_task, task_data)


if __name__ == '__main__':
    service = VideoGenerationService(args)  # Create service with 3 workers
    service.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        service.stop()