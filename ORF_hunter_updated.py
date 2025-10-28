# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 19:53:17 2025

updated according to suggestions from valuable suggestions from reviewer
"""

import os

# Standard genetic code
codon_table = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
}

def reverse_complement(sequence):
    """Generate reverse complement of DNA sequence"""
    complement = {'A':'T', 'T':'A', 'G':'C', 'C':'G', 'N':'N'}
    return ''.join(complement.get(base, '') for base in reversed(sequence.upper()))

def translate_sequence(sequence, frame):
    """Translate DNA sequence in given reading frame (0, 1, 2)"""
    protein = []
    for i in range(frame, len(sequence) - 2, 3):
        codon = sequence[i:i+3]
        if len(codon) == 3:
            amino_acid = codon_table.get(codon.upper(), 'X')
            protein.append(amino_acid)
    return ''.join(protein)

def extract_dna_sequence(full_sequence, start_pos, end_pos, direction):
    """Extract DNA sequence based on coordinates and direction"""
    if direction == 'forward':
        return full_sequence[start_pos:end_pos]
    else:  # reverse
        # For reverse strand, extract from original sequence and return as is
        # The coordinates are already adjusted for the reverse complement
        return full_sequence[start_pos:end_pos]

def find_orfs_six_frame(sequence, min_aa_length=11, max_aa_length=50):
    """Find all ORFs using six-frame translation approach"""
    all_orfs = []
    
    # Process three forward frames
    for frame in range(3):
        protein_seq = translate_sequence(sequence, frame)
        orfs = extract_orfs_from_protein(sequence, protein_seq, frame, 'forward', min_aa_length, max_aa_length)
        all_orfs.extend(orfs)
    
    # Process three reverse frames
    rev_comp = reverse_complement(sequence)
    for frame in range(3):
        protein_seq = translate_sequence(rev_comp, frame)
        orfs = extract_orfs_from_protein(sequence, protein_seq, frame, 'reverse', min_aa_length, max_aa_length)
        all_orfs.extend(orfs)
    
    return all_orfs

def extract_orfs_from_protein(original_sequence, protein_seq, frame, direction, min_length, max_length):
    """Extract ORFs from translated protein sequence"""
    orfs = []
    start_positions = []
    
    # Find all potential start codons (M) and stop codons (*)
    for i, aa in enumerate(protein_seq):
        if aa == 'M':  # Start codon
            start_positions.append(i)
        elif aa == '*':  # Stop codon
            # Check if we have any start positions before this stop
            valid_starts = [start for start in start_positions if start < i]
            for start in valid_starts:
                orf_length = i - start
                if min_length <= orf_length <= max_length:
                    orf_seq = protein_seq[start:i]
                    
                    # Calculate DNA coordinates
                    dna_start = start * 3 + frame
                    dna_end = i * 3 + frame + 3  # Include stop codon
                    
                    if direction == 'reverse':
                        # For reverse strand, adjust coordinates to original sequence
                        seq_len = len(original_sequence)
                        dna_start_rev = seq_len - (start * 3 + frame + 3)
                        dna_end_rev = seq_len - (i * 3 + frame)
                        dna_start, dna_end = dna_start_rev, dna_end_rev
                    
                    # Extract DNA sequence
                    dna_sequence = extract_dna_sequence(original_sequence, dna_start, dna_end, direction)
                    
                    orfs.append({
                        'protein_sequence': orf_seq,
                        'dna_sequence': dna_sequence,
                        'length': orf_length,
                        'frame': frame,
                        'direction': direction,
                        'protein_start': start,
                        'protein_end': i,
                        'dna_start': dna_start,
                        'dna_end': dna_end
                    })
            start_positions = []  # Reset start positions after stop codon
    
    return orfs

def process_fasta_file(input_file, output_csv):
    """Process FASTA file and extract ORFs"""
    sequences = {}
    current_header = ""
    
    # Read and parse FASTA file
    with open(input_file, 'r') as infile:
        for line in infile:
            line = line.strip()
            if line.startswith(">"):
                current_header = line
                sequences[current_header] = ""
            else:
                if "N" not in line.upper():  # Skip lines with ambiguous bases
                    sequences[current_header] += line.upper()
    
    # Process each sequence and find ORFs
    results = []
    for header, sequence in sequences.items():
        orfs = find_orfs_six_frame(sequence)
        for orf in orfs:
            results.append({
                'header': header.strip(),
                'protein_sequence': orf['protein_sequence'],
                'dna_sequence': orf['dna_sequence'],
                'length': orf['length'],
                'frame': orf['frame'],
                'direction': orf['direction'],
                'dna_coords': f"{orf['dna_start']}-{orf['dna_end']}"
            })
    
    # Write results to CSV
    with open(output_csv, 'w') as outfile:
        outfile.write("Header,Protein_Sequence,DNA_Sequence,Length_AA,Frame,Direction,DNA_Coordinates\n")
        for result in results:
            outfile.write(f"{result['header']},{result['protein_sequence']},{result['dna_sequence']},{result['length']},{result['frame']},{result['direction']},{result['dna_coords']}\n")
    
    print(f"Processed {len(sequences)} sequences, found {len(results)} ORFs")
    return results

# Example usage
if __name__ == "__main__":
    input_fasta = "your_input_sequence.fasta"  # Replace with your input file
    output_file = "orf_results.csv"  # Replace with your output file
    
    results = process_fasta_file(input_fasta, output_file)
