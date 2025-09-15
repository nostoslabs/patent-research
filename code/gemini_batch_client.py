"""Gemini Batch API client for ground truth generation."""

import json
import time
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BatchJob:
    """Represents a batch job."""
    job_id: str
    name: str
    state: str
    create_time: str
    update_time: str
    request_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    completion_percentage: float = 0.0


class GeminiBatchClient:
    """Client for Gemini Batch API operations."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize batch client with API key."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
        
        self.base_url = "https://generativelanguage.googleapis.com"
        self.headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        
        logger.info("Gemini Batch API client initialized")
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make authenticated API request."""
        url = f"{self.base_url}{endpoint}"
        
        response = requests.request(method, url, headers=self.headers, **kwargs)
        
        if response.status_code not in [200, 201]:
            logger.error(f"API request failed: {response.status_code} - {response.text}")
            response.raise_for_status()
        
        return response
    
    def submit_batch(self, 
                    input_file: str, 
                    job_name: str = None,
                    model: str = "gemini-1.5-flash") -> BatchJob:
        """Submit a batch job from JSONL file."""
        if not Path(input_file).exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        if not job_name:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            job_name = f"ground_truth_batch_{timestamp}"
        
        logger.info(f"Submitting batch job: {job_name}")
        logger.info(f"Input file: {input_file}")
        
        # Read and prepare requests
        requests_data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    request_data = json.loads(line.strip())
                    requests_data.append(request_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON on line {line_num}: {e}")
                    continue
        
        if not requests_data:
            raise ValueError("No valid requests found in input file")
        
        logger.info(f"Loaded {len(requests_data)} requests")
        
        # Create batch request payload using inline requests for small batches
        batch_payload = {
            "displayName": job_name,
            "inlineRequests": [req["request"] for req in requests_data]
        }
        
        # Submit batch job using the correct endpoint format
        endpoint = f"/v1beta/models/{model}:batchGenerateContent"
        
        try:
            response = self._make_request("POST", endpoint, json=batch_payload)
            batch_data = response.json()
            
            # For inline requests, the response is immediate
            if "responses" in batch_data:
                # Process immediate response and save to a pseudo-job format
                job_id = f"inline_{job_name}_{int(time.time())}"
                
                # Save immediate results
                results_file = f"{job_name}_immediate_results.json"
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(batch_data, f, indent=2, ensure_ascii=False)
                
                job = BatchJob(
                    job_id=job_id,
                    name=job_name,
                    state="STATE_COMPLETED",
                    create_time=time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    update_time=time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    request_count=len(requests_data),
                    completed_count=len(batch_data.get("responses", [])),
                    failed_count=0,
                    completion_percentage=100.0
                )
                
                logger.info(f"Inline batch completed immediately!")
                logger.info(f"Results saved to: {results_file}")
                
            else:
                # Async batch job (for larger batches)
                job = BatchJob(
                    job_id=batch_data["name"],
                    name=job_name,
                    state=batch_data.get("state", "UNKNOWN"),
                    create_time=batch_data.get("createTime", ""),
                    update_time=batch_data.get("updateTime", ""),
                    request_count=len(requests_data)
                )
            
            logger.info(f"Batch job submitted successfully!")
            logger.info(f"Job ID: {job.job_id}")
            logger.info(f"State: {job.state}")
            logger.info(f"Request count: {job.request_count}")
            
            # Save job info
            self._save_job_info(job, input_file)
            
            return job
            
        except Exception as e:
            logger.error(f"Failed to submit batch job: {e}")
            raise
    
    def get_batch_status(self, job_id: str) -> BatchJob:
        """Get status of a batch job."""
        endpoint = f"/v1beta/{job_id}"
        
        try:
            response = self._make_request("GET", endpoint)
            batch_data = response.json()
            
            # Parse batch statistics
            stats = batch_data.get("batchStats", {})
            completed = stats.get("completedRequestsCount", 0)
            failed = stats.get("failedRequestsCount", 0)
            total = stats.get("totalRequestsCount", 0)
            
            completion_percentage = (completed + failed) / total * 100 if total > 0 else 0
            
            job = BatchJob(
                job_id=batch_data["name"],
                name=job_id.split("/")[-1],
                state=batch_data.get("state", "UNKNOWN"),
                create_time=batch_data.get("createTime", ""),
                update_time=batch_data.get("updateTime", ""),
                request_count=total,
                completed_count=completed,
                failed_count=failed,
                completion_percentage=completion_percentage
            )
            
            return job
            
        except Exception as e:
            logger.error(f"Failed to get batch status: {e}")
            raise
    
    def download_results(self, job_id: str, output_file: str) -> str:
        """Download batch job results."""
        # First check if job is complete
        job = self.get_batch_status(job_id)
        
        if job.state not in ["STATE_SUCCEEDED", "STATE_COMPLETED"]:
            logger.warning(f"Job is not complete yet. State: {job.state}")
            if job.state == "STATE_FAILED":
                raise RuntimeError(f"Batch job failed: {job_id}")
        
        endpoint = f"/v1beta/{job_id}/results"
        
        try:
            response = self._make_request("GET", endpoint)
            results_data = response.json()
            
            # Save raw results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results downloaded to: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to download results: {e}")
            raise
    
    def process_results(self, 
                       results_file: str, 
                       output_file: str) -> Dict[str, Any]:
        """Process raw batch results into ground truth format."""
        logger.info(f"Processing results from: {results_file}")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        # Extract individual results
        responses = results_data.get("responses", [])
        if not responses:
            raise ValueError("No responses found in results file")
        
        processed_results = []
        successful_count = 0
        failed_count = 0
        
        for response in responses:
            try:
                # Extract metadata from original request key
                key = response.get("key", "")
                
                # Check if response was successful
                if "error" in response:
                    logger.warning(f"Failed response for {key}: {response['error']}")
                    failed_count += 1
                    continue
                
                # Extract the LLM response
                candidates = response.get("response", {}).get("candidates", [])
                if not candidates:
                    logger.warning(f"No candidates in response for {key}")
                    failed_count += 1
                    continue
                
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if not parts:
                    logger.warning(f"No content parts in response for {key}")
                    failed_count += 1
                    continue
                
                # Parse JSON response
                llm_response_text = parts[0].get("text", "")
                try:
                    llm_analysis = json.loads(llm_response_text)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON response for {key}: {e}")
                    failed_count += 1
                    continue
                
                # Extract patent IDs from key
                if key.startswith("pair_"):
                    parts = key[5:].split("_", 1)  # Remove "pair_" prefix
                    if len(parts) == 2:
                        patent1_id, patent2_id = parts
                    else:
                        logger.warning(f"Could not parse patent IDs from key: {key}")
                        patent1_id = patent2_id = "unknown"
                else:
                    patent1_id = patent2_id = "unknown"
                
                # Create processed result
                result = {
                    'patent1_id': patent1_id,
                    'patent2_id': patent2_id,
                    'llm_analysis': llm_analysis,
                    'success': True,
                    'error': None,
                    'batch_key': key
                }
                
                processed_results.append(result)
                successful_count += 1
                
            except Exception as e:
                logger.error(f"Error processing response: {e}")
                failed_count += 1
                continue
        
        # Save processed results
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in processed_results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        
        # Generate summary
        summary = {
            'total_responses': len(responses),
            'successful_evaluations': successful_count,
            'failed_evaluations': failed_count,
            'success_rate': successful_count / len(responses) if responses else 0,
            'processed_file': output_file,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save summary
        summary_file = output_file.replace('.jsonl', '_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results processing completed!")
        logger.info(f"Successful evaluations: {successful_count}/{len(responses)}")
        logger.info(f"Success rate: {summary['success_rate']:.1%}")
        logger.info(f"Processed results saved to: {output_file}")
        logger.info(f"Summary saved to: {summary_file}")
        
        return summary
    
    def _save_job_info(self, job: BatchJob, input_file: str):
        """Save job information for tracking."""
        job_info = {
            'job_id': job.job_id,
            'name': job.name,
            'state': job.state,
            'create_time': job.create_time,
            'request_count': job.request_count,
            'input_file': input_file,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        job_file = f"batch_job_{job.name}.json"
        with open(job_file, 'w', encoding='utf-8') as f:
            json.dump(job_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Job info saved to: {job_file}")


def main():
    """CLI interface for batch operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gemini Batch API client")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Submit a batch job')
    submit_parser.add_argument('input_file', help='Input JSONL file with batch requests')
    submit_parser.add_argument('--name', help='Job name (auto-generated if not provided)')
    submit_parser.add_argument('--model', default='gemini-1.5-flash', help='Model to use')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check batch job status')
    status_parser.add_argument('job_id', help='Batch job ID')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download batch results')
    download_parser.add_argument('job_id', help='Batch job ID')
    download_parser.add_argument('output_file', help='Output file for results')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process raw results')
    process_parser.add_argument('results_file', help='Raw results file')
    process_parser.add_argument('output_file', help='Processed output file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        client = GeminiBatchClient()
        
        if args.command == 'submit':
            job = client.submit_batch(args.input_file, args.name, args.model)
            print(f"Job submitted: {job.job_id}")
            
        elif args.command == 'status':
            job = client.get_batch_status(args.job_id)
            print(f"Job ID: {job.job_id}")
            print(f"State: {job.state}")
            print(f"Progress: {job.completion_percentage:.1f}%")
            print(f"Completed: {job.completed_count}/{job.request_count}")
            print(f"Failed: {job.failed_count}")
            
        elif args.command == 'download':
            output_file = client.download_results(args.job_id, args.output_file)
            print(f"Results downloaded to: {output_file}")
            
        elif args.command == 'process':
            summary = client.process_results(args.results_file, args.output_file)
            print(f"Processing completed: {summary['success_rate']:.1%} success rate")
            
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        raise


if __name__ == "__main__":
    main()