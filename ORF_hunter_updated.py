# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 19:53:17 2025

Updated ORF prediction - An Improved version
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
        extracted = full_sequence[start_pos:end_pos]
        return reverse_complement(extracted)

def check_met_retention(protein_sequence):
    """
    Check if the initial Met should be retained based on P1' position.
    Met is retained when P1' position has bulky residues: I, Y, E, R.
    Returns: (should_retain_met, p1_position_aa)
    """
    if len(protein_sequence) < 2:
        return False, None
    
    p1_position_aa = protein_sequence[1]  # Second amino acid (P1' position)
    
    # MetAP does not cleave when P1' has bulky residues
    bulky_residues = {'I', 'Y', 'E', 'R'}  # Ile, Tyr, Glu, Arg
    should_retain = p1_position_aa in bulky_residues
    
    return should_retain, p1_position_aa

def process_protein_sequence(protein_sequence, met_retained):
    """
    Process protein sequence based on Met retention rule.
    If Met should not be retained, remove the initial M.
    """
    if protein_sequence.startswith('M') and not met_retained:
        return protein_sequence[1:]
    else:
        return protein_sequence

def find_orfs_six_frame(sequence, min_aa_length=11, max_aa_length=50):
    """Find all ORFs using improved six-frame translation approach"""
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
    """Extract ORFs from translated protein sequence - IMPROVED VERSION"""
    orfs = []
    start_positions = []
    
    # Iterate through the protein sequence
    for i, aa in enumerate(protein_seq):
        if aa == 'M':  # Start codon
            start_positions.append(i)
        elif aa == '*':  # Stop codon
            # Check all start positions before this stop
            valid_starts = [start for start in start_positions if start < i]
            
            for start in valid_starts:
                orf_length = i - start
                if min_length <= orf_length <= max_length:
                    original_orf_seq = protein_seq[start:i]
                    
                    # Check Met retention rule for AMPs
                    met_retained, p1_aa = check_met_retention(original_orf_seq)
                    
                    # Process protein sequence based on Met retention
                    processed_protein_seq = process_protein_sequence(original_orf_seq, met_retained)
                    processed_length = len(processed_protein_seq)
                    
                    # Calculate DNA coordinates
                    if direction == 'forward':
                        dna_start = start * 3 + frame
                        dna_end = i * 3 + frame + 3  # Include stop codon
                    else:  # reverse
                        seq_len = len(original_sequence)
                        # Correct coordinate calculation for reverse strand
                        dna_start = seq_len - (i * 3 + frame + 3)
                        dna_end = seq_len - (start * 3 + frame)
                    
                    # Ensure coordinates are within bounds and start < end
                    dna_start = max(0, dna_start)
                    dna_end = min(len(original_sequence), dna_end)
                    if dna_start >= dna_end:
                        continue
                    
                    # Extract DNA sequence
                    dna_sequence = extract_dna_sequence(original_sequence, dna_start, dna_end, direction)
                    
                    orfs.append({
                        'original_protein_sequence': original_orf_seq,
                        'protein_sequence': processed_protein_seq,
                        'dna_sequence': dna_sequence,
                        'original_length': orf_length,
                        'processed_length': processed_length,
                        'frame': frame,
                        'direction': direction,
                        'protein_start': start,
                        'protein_end': i,
                        'dna_start': dna_start,
                        'dna_end': dna_end,
                        'met_retained': met_retained,
                        'p1_position_aa': p1_aa,
                        'has_initial_met': original_orf_seq.startswith('M'),
                        'm_removed': original_orf_seq.startswith('M') and not met_retained
                    })
            
            # CRITICAL FIX: Only remove the starts that were used, not all starts
            # Remove only the start positions that are before this stop codon
            start_positions = [start for start in start_positions if start > i]
    
    # IMPORTANT: Also check for ORFs that continue until the end of sequence (no stop codon)
    for start in start_positions:
        orf_length = len(protein_seq) - start
        if min_length <= orf_length <= max_length:
            original_orf_seq = protein_seq[start:]
            
            met_retained, p1_aa = check_met_retention(original_orf_seq)
            processed_protein_seq = process_protein_sequence(original_orf_seq, met_retained)
            processed_length = len(processed_protein_seq)
            
            if direction == 'forward':
                dna_start = start * 3 + frame
                dna_end = len(original_sequence)  # Go to end of sequence
            else:
                seq_len = len(original_sequence)
                dna_start = seq_len - (len(protein_seq) * 3 + frame)
                dna_end = seq_len - (start * 3 + frame)
            
            dna_start = max(0, dna_start)
            dna_end = min(len(original_sequence), dna_end)
            if dna_start >= dna_end:
                continue
                
            dna_sequence = extract_dna_sequence(original_sequence, dna_start, dna_end, direction)
            
            orfs.append({
                'original_protein_sequence': original_orf_seq,
                'protein_sequence': processed_protein_seq,
                'dna_sequence': dna_sequence,
                'original_length': orf_length,
                'processed_length': processed_length,
                'frame': frame,
                'direction': direction,
                'protein_start': start,
                'protein_end': len(protein_seq),
                'dna_start': dna_start,
                'dna_end': dna_end,
                'met_retained': met_retained,
                'p1_position_aa': p1_aa,
                'has_initial_met': original_orf_seq.startswith('M'),
                'm_removed': original_orf_seq.startswith('M') and not met_retained
            })
    
    return orfs

def process_fasta_file(input_file, output_csv):
    """Process FASTA file and extract ORFs with Met retention analysis"""
    sequences = {}
    current_header = ""
    
    with open(input_file, 'r') as infile:
        for line in infile:
            line = line.strip()
            if line.startswith(">"):
                current_header = line
                sequences[current_header] = ""
            else:
                sequences[current_header] += line.upper()
    
    results = []
    met_retention_count = 0
    total_orfs_with_met = 0
    m_removed_count = 0
    
    for header, sequence in sequences.items():
        orfs = find_orfs_six_frame(sequence)
        for orf in orfs:
            if orf['has_initial_met']:
                total_orfs_with_met += 1
                if orf['met_retained']:
                    met_retention_count += 1
                if orf['m_removed']:
                    m_removed_count += 1
            
            results.append({
                'header': header.strip(),
                'protein_sequence': orf['protein_sequence'],
                'dna_sequence': orf['dna_sequence'],
                'original_length': orf['original_length'],
                'processed_length': orf['processed_length'],
                'frame': orf['frame'],
                'direction': orf['direction'],
                'dna_coords': f"{orf['dna_start']}-{orf['dna_end']}",
                'met_retained': orf['met_retained'],
                'p1_position_aa': orf['p1_position_aa'],
                'has_initial_met': orf['has_initial_met'],
                'm_removed': orf['m_removed']
            })
    
    with open(output_csv, 'w') as outfile:
        outfile.write("Header,Protein_Sequence,DNA_Sequence,Original_Length_AA,Processed_Length_AA,Frame,Direction,DNA_Coordinates,Has_Initial_Met,P1_Position_AA,Met_Retained,M_Removed\n")
        for result in results:
            outfile.write(f"{result['header']},{result['protein_sequence']},{result['dna_sequence']},{result['original_length']},{result['processed_length']},{result['frame']},{result['direction']},{result['dna_coords']},{result['has_initial_met']},{result['p1_position_aa']},{result['met_retained']},{result['m_removed']}\n")
    
    print(f"Processed {len(sequences)} sequences, found {len(results)} ORFs")
    print(f"ORFs with initial Met: {total_orfs_with_met}")
    print(f"ORFs with retained Met (P1' = I/Y/E/R): {met_retention_count}")
    print(f"ORFs with M removed (P1' ≠ I/Y/E/R): {m_removed_count}")
    print("Note: Initial Met residues are preserved when P1' position has bulky residues (Ile, Tyr, Glu, Arg); References: Protein N-terminal processing: substrate specificity of Escherichia coli and human methionine aminopeptidases. Biochemistry. 2010; 49(26): 5588–99. https://doi.org/10.1021/bi1005464")
    
    return results

# Example usage
if __name__ == "__main__":
    input_fasta = "your_input_sequence.fasta"  # Replace with your input file
    output_file = "orf_results_improved.csv"  # Replace with your output file
   
    results = process_fasta_file(input_fasta, output_file)
