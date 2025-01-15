import os
import random
import argparse

def sample_lines_from_files(input_files, output_file, num_lines):
    all_sampled_lines = []
    num_files = len(input_files)
    lines_per_file = num_lines // num_files
    
    for file_path in input_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) < lines_per_file:
                    print(f"Warning: File {file_path} has only {len(lines)} lines, but {lines_per_file} lines requested.")
                    all_sampled_lines.extend(lines)
                else:
                    all_sampled_lines.extend(random.sample(lines, lines_per_file))
        except IOError as e:
            print(f"Error reading file {file_path}: {e}")
            continue
    
    if len(all_sampled_lines) < num_lines:
        print(f"Warning: Requested {num_lines} lines, but only {len(all_sampled_lines)} lines available.")
    
    random.shuffle(all_sampled_lines)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(all_sampled_lines[:num_lines])
        print(f"Successfully created {output_file} with {len(all_sampled_lines[:num_lines])} lines.")
    except IOError as e:
        print(f"Error writing to output file {output_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Sample random lines from multiple input files.')
    parser.add_argument('input_files', nargs='+', help='Input file paths')
    parser.add_argument('-o', '--output', default='training_dataset.txt', 
                        help='Output file path (default: training_dataset.txt)')
    parser.add_argument('-n', '--num_lines', type=int, default=1000, 
                        help='Number of lines to sample (default: 1000)')
    
    args = parser.parse_args()
    
    sample_lines_from_files(args.input_files, args.output, args.num_lines)

if __name__ == '__main__':
    main()