import exiftool
import os
import arrow
import logging
import argparse
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhotoProcessor:
    """Processes photos to extract date metadata and generate new file paths."""
    
    def __init__(self, directory: str, output_file: str, nocreationdate_file: str):
        self.directory = directory
        self.output_file = output_file
        self.nocreationdate_file = nocreationdate_file
        self.ignore_files = ['.DS_Store', 'test.out', 'nocreationdate.txt', 'output.txt']
        self.custom_formats = ['YYYY:MM:DD HH:mm:ssZZ', 'YYYY:MM:DD HH:mm:ss', 'YYYY:MM:DD']
        self.counter_dict: Dict[str, Dict[str, Dict[str, int]]] = {}
    
    @contextmanager
    def open_output_files(self):
        """Context manager for safely opening and closing output files."""
        try:
            with open(self.output_file, 'w') as outfile, \
                 open(self.nocreationdate_file, 'w') as errorfile:
                yield outfile, errorfile
        except IOError as e:
            logger.error(f"Error opening output files: {e}")
            raise
    
    def extract_earliest_date(self, metadata: Dict) -> Optional[arrow.Arrow]:
        """Extract the earliest date from photo metadata."""
        earliest_date = None
        
        for key, value in metadata.items():
            if "date" in key.lower():
                try:
                    parsed_date = arrow.get(value, self.custom_formats, normalize_whitespace=True)
                    logger.debug(f"Date: {value} -> {parsed_date}")
                    
                    if earliest_date is None or parsed_date.timestamp() < earliest_date.timestamp():
                        earliest_date = parsed_date
                        
                except arrow.ParserError:
                    logger.debug(f"Parse Error for date: {value}")
                    continue
                except (ValueError, TypeError) as e:
                    logger.debug(f"Date parsing error: {e} for {value}")
                    continue
        
        return earliest_date
    
    def parse_path_components(self, file_path: str) -> Tuple[str, str, str]:
        """Extract year, month, and day from file path."""
        root_parts = os.path.dirname(file_path).split(os.path.sep)
        
        # Handle case where path doesn't have expected structure
        if len(root_parts) < 2:
            logger.warning(f"Path structure unexpected for {file_path}, using defaults")
            return "unknown", "unknown", "unknown"
        
        year = root_parts[-2] if len(root_parts) >= 2 else "unknown"
        
        # Handle case where last part doesn't contain a dash
        last_part = root_parts[-1]
        if "-" in last_part:
            month, day = last_part.split("-", 1)
        else:
            logger.warning(f"Directory name doesn't contain expected format for {file_path}")
            month, day = "unknown", "unknown"
            
        return year, month, day
    
    def generate_dated_filename(self, directory: str, year: str, month: str, day: str, 
                              date: arrow.Arrow, extension: str) -> str:
        """Generate filename with date timestamp."""
        return os.path.join(
            directory, year, f"{month}-{day}", 
            f"{date.datetime.isoformat()}{extension.lower()}"
        )
    
    def generate_counter_filename(self, directory: str, year: str, month: str, day: str, 
                                extension: str) -> str:
        """Generate filename with counter for files without dates."""
        if year not in self.counter_dict:
            self.counter_dict[year] = {}
        if month not in self.counter_dict[year]:
            self.counter_dict[year][month] = {}
        if day not in self.counter_dict[year][month]:
            self.counter_dict[year][month][day] = 0
        else:
            self.counter_dict[year][month][day] += 1
        
        counter = self.counter_dict[year][month][day]
        return os.path.join(
            directory, year, f"{month}-{day}", 
            f"{counter}{extension.lower()}"
        )
    
    def process_file(self, file_path: str, et: exiftool.ExifToolHelper, 
                    outfile, errorfile) -> None:
        """Process a single file to extract date and generate new path."""
        try:
            year, month, day = self.parse_path_components(file_path)
            extension = os.path.splitext(file_path)[1]
            
            metadata_list = et.get_metadata(file_path)
            if not metadata_list:
                logger.warning(f"No metadata found for {file_path}")
                return
            
            earliest_date = self.extract_earliest_date(metadata_list[0])
            
            if earliest_date is not None:
                new_filename = self.generate_dated_filename(
                    self.directory, year, month, day, earliest_date, extension
                )
                outfile.write(f"{file_path} -> {new_filename}\n")
                logger.debug(f"Processed with date: {file_path} -> {new_filename}")
            else:
                new_filename = self.generate_counter_filename(
                    self.directory, year, month, day, extension
                )
                errorfile.write(f"{file_path} -> {new_filename}\n")
                logger.debug(f"Processed without date: {file_path} -> {new_filename}")
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    def process_directory(self) -> None:
        """Process all files in the directory."""
        if not os.path.exists(self.directory):
            logger.error(f"Directory {self.directory} not mounted")
            return
        
        try:
            with self.open_output_files() as (outfile, errorfile):
                with exiftool.ExifToolHelper() as et:
                    file_count = 0
                    
                    for root, dirs, files in os.walk(self.directory):
                        for filename in files:
                            if filename not in self.ignore_files:
                                full_file_path = os.path.join(root, filename)
                                self.process_file(full_file_path, et, outfile, errorfile)
                                file_count += 1
                                
                                if file_count % 100 == 0:
                                    logger.info(f"Processed {file_count} files...")
                    
                    logger.info(f"Processing complete. Total files processed: {file_count}")
                    
        except exiftool.exceptions.ExifToolExecuteError as e:
            logger.error(f"ExifTool error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Process photos and extract date metadata')
    parser.add_argument('--directory', '-d', default='/Volumes/Shared/Lightroom',
                       help='Directory containing photos to process')
    parser.add_argument('--output', '-o', default='./output.txt',
                       help='Output file for successful date extractions')
    parser.add_argument('--nocreationdate', '-n', default='./nocreationdate.txt',
                       help='Output file for files without creation dates')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    processor = PhotoProcessor(args.directory, args.output, args.nocreationdate)
    processor.process_directory()


if __name__ == '__main__':
    main()
