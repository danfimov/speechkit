import asyncio
import base64
import pathlib
import random
import time
import uuid
from collections import defaultdict
from statistics import mean

import anyio
import click
import httpx
import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table


console = Console()


def get_audio_files(directory: str) -> list[str]:
    """Получаем список всех аудио файлов в директории"""
    audio_extensions = {'.mp3', '.wav', '.ogg', '.m4a', '.flac'}
    files = [str(file) for file in pathlib.Path(directory).rglob('*') if file.suffix.lower() in audio_extensions]
    if not files:
        msg = f'No audio files found in {directory}'
        raise click.BadParameter(msg)
    return files


async def make_request(client: httpx.AsyncClient, file_path: str) -> dict:
    url = 'https://speechkit-app.query.consul-test/api/v1/recognition/'
    params = {'model_name': 'whisper'}

    auth_string = '12345:12345'
    auth_base64 = base64.b64encode(auth_string.encode('ascii')).decode('ascii')
    headers = {'accept': 'application/json', 'authorization': f'Basic {auth_base64}', 'x-request-id': str(uuid.uuid4())}

    start_time = time.time()
    try:
        file_path_obj = pathlib.Path(file_path)
        async with await anyio.open_file(file_path, 'rb') as f:
            file_content = await f.read()
            files = {'file': (file_path_obj.name, file_content, 'audio/mpeg')}

            response = await client.post(url, data=params, headers=headers, files=files)
    except Exception as e:  # noqa: BLE001
        duration = time.time() - start_time
        return {'status': 0, 'duration': duration, 'error': str(e), 'file': file_path}
    else:
        duration = time.time() - start_time
        return {'status': response.status_code, 'duration': duration, 'error': None, 'file': file_path}


async def run_batch(files: list[str], concurrent_requests: int, progress, task_id) -> list[dict]:
    async with httpx.AsyncClient(timeout=30.0, verify=False) as client:  # noqa: S501
        batch_files = random.choices(files, k=concurrent_requests)  # noqa: S311
        tasks = [make_request(client, file) for file in batch_files]
        results = []
        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            results.append(result)
            progress.update(task_id, advance=1)
        return results


def print_ascii_histogram(durations: list[float], bins: int = 10, width: int = 40) -> None:
    """Prints ASCII histogram of response times in the console"""
    if not durations:
        return

    min_dur = min(durations)
    max_dur = max(durations)
    bin_size = (max_dur - min_dur) / bins if max_dur > min_dur else 1.0

    histogram = defaultdict(int)
    for d in durations:
        bin_idx = min(int((d - min_dur) / bin_size), bins - 1)
        histogram[bin_idx] += 1

    max_count = max(histogram.values()) if histogram else 0
    console.print('\n[bold]Response time histogram:[/bold]')
    for i in range(bins):
        bin_start = min_dur + i * bin_size
        count = histogram[i]
        bar_length = int((count / max_count) * width) if max_count > 0 else 0
        bar = '■' * bar_length
        console.print(f'{bin_start:6.3f} [{count:3d}] |{bar}')


def print_statistics(results: list[dict]) -> None:
    status_counts = defaultdict(int)
    durations = []
    errors = []
    files_stats = defaultdict(list)

    for r in results:
        status_counts[r['status']] += 1
        durations.append(r['duration'])
        if r['error']:
            errors.append(r['error'])
        files_stats[r['file']].append(r['duration'])

    # Table of status codes
    status_table = Table(title='Response Status Codes')
    status_table.add_column('Status Code')
    status_table.add_column('Count')
    status_table.add_column('Percentage')

    total = len(results)
    for status, count in status_counts.items():
        percentage = (count / total) * 100
        status_table.add_row(str(status), str(count), f'{percentage:.1f}%')

    # Table of response timing
    timing_table = Table(title='Response Timing (seconds)')
    timing_table.add_column('Metric')
    timing_table.add_column('Value')

    if durations:
        durations_array = np.array(durations)
        percentiles = np.percentile(durations_array, [50, 75, 90, 95, 99])

        timing_table.add_row('Min', f'{min(durations):.3f}')
        timing_table.add_row('Max', f'{max(durations):.3f}')
        timing_table.add_row('Mean', f'{mean(durations):.3f}')
        timing_table.add_row('Median (P50)', f'{percentiles[0]:.3f}')
        timing_table.add_row('P75', f'{percentiles[1]:.3f}')
        timing_table.add_row('P90', f'{percentiles[2]:.3f}')
        timing_table.add_row('P95', f'{percentiles[3]:.3f}')
        timing_table.add_row('P99', f'{percentiles[4]:.3f}')

    # Calculates the statistics for files
    files_table = Table(title='Bullet Statistics')
    files_table.add_column('Bullet')
    files_table.add_column('Requests')
    files_table.add_column('Avg Time')

    for file, times in files_stats.items():
        files_table.add_row(pathlib.Path(file).name, str(len(times)), f'{mean(times):.3f}s')

    # Print statistics
    console.print('\n[bold]Test Results[/bold]')
    console.print(status_table)
    console.print(timing_table)
    console.print(files_table)

    print_ascii_histogram(durations)

    if errors:
        console.print('\n[bold red]Errors:[/bold red]')
        for error in set(errors):
            console.print(f'- {error}')


@click.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--workers', '-w', default=10, help='Number of concurrent workers')
@click.option('--requests', '-n', default=100, help='Total number of requests')
def main(directory: str, workers: int, requests: int) -> None:
    """Load test the speech recognition API"""
    # Gets the list of files
    files = get_audio_files(directory)
    console.print(f'Found {len(files)} audio files in {directory}')
    console.print(f'Starting test with {workers} workers and {requests} total requests')

    # Calculates the number of batches
    batch_size = min(workers, requests)
    num_batches = (requests + batch_size - 1) // batch_size
    last_batch_size = requests % batch_size or batch_size

    all_results = []
    start_time = time.time()

    # Creates progress bar
    with Progress(
        SpinnerColumn(),
        '[progress.description]{task.description}',
        BarColumn(),
        '[progress.percentage]{task.percentage:>3.0f}%',
        TimeElapsedColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task('[cyan]Processing requests...', total=requests)

        for i in range(num_batches):
            current_batch_size = last_batch_size if i == num_batches - 1 else batch_size
            results = asyncio.run(run_batch(files, current_batch_size, progress, task_id))
            all_results.extend(results)

    total_time = time.time() - start_time
    console.print(f'\nTotal test time: {total_time:.2f} seconds')
    console.print(f'Requests per second: {requests / total_time:.2f}')

    print_statistics(all_results)


if __name__ == '__main__':
    main()
