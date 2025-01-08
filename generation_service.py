# ... existing imports ...
import redis
import json
from threading import Thread
import time
from video_generation import VideoGenerator
from pathlib import Path

args = {
    "yolox_config": "./pose/config/yolox_l_8xb8-300e_coco.py",
    "yolox_ckpt": "./pretrained_weights/dwpose/yolox_l_8x8_300e_coco.pth",
    "dwpose_config": "./pose/config/dwpose-l_384x288.py",
    "dwpose_ckpt": "./pretrained_weights/dwpose/dw-ll_ucoco_384.pth",
    "align_frame": 0,
    "max_frame": 300,
    "detect_resolution": 512,
    "image_resolution": 720,
    "H": 512,
    "W": 512,
    "L": 300,
    "S": 48,
    "O": 4,
    "cfg": 3.5,
    "seed": 99,
    "steps": 20,
    "skip": 1,
    "config": "./configs/model_config.yaml"

}

class VideoGenerationService:
    def __init__(self, args):
        self.video_generator = VideoGenerator(args)
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.task_queue = "video_generation_tasks"
        self.result_queue = "video_generation_results"
        self.should_run = True

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
        print("Video Generation Service stopped")

    def _process_queue(self):
        """Main loop to process tasks from the queue"""
        while self.should_run:
            # Get task from Redis queue with timeout
            task = self.redis_client.blpop(self.task_queue, timeout=1)
            
            if task:
                try:
                    # Task format: {ref_image_path: str, ref_video_path: str, task_id: str}
                    _, task_data = task
                    task_dict = json.loads(task_data)
                    
                    print(f"Processing task: {task_dict['task_id']}")
                    
                    # Process the video generation
                    output_path =self.video_generator(
                        task_dict['ref_image_path'],
                        task_dict['ref_video_path'],
                        task_dict['save_dir']
                    )
                    
                    # Send result
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


if __name__ == '__main__':
    service = VideoGenerationService(args)
    service.start()
    
    # Keep the main thread running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        service.stop()