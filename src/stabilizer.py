def pauli_mult(a, b):
    if a == 'I':
        return [b,1]
    elif b == 'I':
        return [a,1]
    elif a == b:
        return ['I',1]
    elif a == 'X':
        return ['Y',complex(0,-1)] if b == 'Z' else ['Z',complex(0,1)]
    elif a == 'Y':
        return ['X',complex(0,1)] if b == 'Z' else ['Z',complex(0,-1)]
    elif a == 'Z':
        return ['Y',complex(0,1)] if b == 'X' else ['X',complex(0,-1)]

def ROWMULT(M, i, j):
    result = []
    phase = M[i][-1] * M[j][-1]  # Multiply phase factors
    for a, b in zip(M[i][:-1], M[j][:-1]):  # Exclude the phase in the multiplication
        p_mult=pauli_mult(a, b)
        result.append(p_mult[0])
        phase=phase*p_mult[1]
    
    result.append(phase)
    return result

def ROWSWAP(M, i, j):
    M[i], M[j] = M[j], M[i]


###pass in a tableau in the form [["X","Y"],["I","Z"]] and it do gaussian decomposition to reduce to canonical form. 
### can generate from a qiskit stabilizer state:
# for n in stab_state.clifford.to_labels(mode="S"):
#     row=[]
#     for m in n[:]:
#         row.append(m)
#     stab_matrix.append(row[1:]+[1 if row[0]=="+" else -1 ])
# reduced_stab_matrix=stabilizer.reduce_to_row_echelon(stab_matrix)
def reduce_to_row_echelon(M):
    ### https://arxiv.org/pdf/1711.07848 ###
    n = len(M)
    i = 0

    # Setup X block
    for j in range(n):
        k = next((k for k in range(i, n) if M[k][j] in {'X', 'Y'}), None)
        if k is not None:
            ROWSWAP(M, i, k)
            for m in range(n):
                if m != i and M[m][j] in {'X', 'Y'}:
                    M[m] = ROWMULT(M,i, m)
            i += 1
    
    # Setup Z block
    for j in range(n):
        k = next((k for k in range(i, n) if M[k][j] == 'Z'), None)
        if k is not None:
            ROWSWAP(M, i, k)
            for m in range(n):
                if m != i and M[m][j] in {'Z', 'Y'}:
                    M[m] = ROWMULT(M,i, m)
            i += 1

    return M

###apply a Pauli string to a basis state
def apply_stab_mat_row(pre_state, row,phase):
    post_state = pre_state
    phase=phase * row[-1]
    for n in range(len(post_state)):
        if post_state[n] == "1":
            if row[n] == "X":
                post_state = post_state[:n]+"0"+post_state[n+1:]
                phase = phase * 1
            elif row[n] == "Y":
                post_state = post_state[:n]+"0"+post_state[n+1:]
                phase=phase * complex(0,-1)
            elif row[n] == "Z":
                post_state = post_state[:n]+"1"+post_state[n+1:]
                phase = phase * -1
            elif row[n] == "I":
                post_state = post_state[:n]+"1"+post_state[n+1:]
        elif post_state[n] == "0":   
            if row[n] == "X":
                post_state = post_state[:n]+"1"+post_state[n+1:]
                phase = phase * 1
            elif row[n] == "Y":
                post_state = post_state[:n]+"1"+post_state[n+1:]
                phase=phase * complex(0,1)
            elif row[n] == "Z":
                post_state = post_state[:n]+"0"+post_state[n+1:]
                phase = phase * 1
            elif row[n] == "I":
                post_state = post_state[:n]+"0"+post_state[n+1:]             

    return post_state, phase


### find phase of the overlap between a one stabilizer (reduced stab matrix) and a basis state (ovlp_state),
### requires one of the basis states that exist in the stabilizer to be passed as ref_state, then searches for sequence of Paulis from reduced tableau that will transform to the basis state of interest 
### e.g. ref_state=stab_state.measure()[0] for qiskit stabilizer state
def get_ovlp_phase_basis(ref_state,ovlp_state,reduced_stab_matrix):
    #ref state is one of the computational basis states that exist in the stabilizer represented by reduced stab matrix. ovlp_state is a computational basis state 
    phase=1
    interm_state=ref_state
    for n in range(len(ovlp_state)):
        #print(ovlp_state,interm_state,phase)
        #print(ovlp_state[n],interm_state[n])
        if ovlp_state[n]!= interm_state[n]:
            applied_q=False
            for m in range(len(ovlp_state)):
                if (reduced_stab_matrix[m][n]=="Y" or reduced_stab_matrix[m][n]=="X") and ("X" not in reduced_stab_matrix[m][:n]) and ("Y" not in reduced_stab_matrix[m][:n]):
            #if (reduced_stab_matrix[n][n]=="X" or reduced_stab_matrix[n][n]=="Y"): #if there exists a row with X or Y on the furthest left non I or Z element for the column of interest
                    interm_state, phase = apply_stab_mat_row(interm_state,reduced_stab_matrix[m],phase)
                    applied_q=True
            if not applied_q:
                phase=0.   
    return phase    


##find the number of generators containing an X or Y Pauli in a reduced tableau
def x_rank_of_reduced(reduced_M):
    for m in range(len(reduced_M)):
        if "X" not in reduced_M[m] and "Y" not in reduced_M[m]:
            return m
        
    return len(reduced_M)


def get_ovlp(reduced_stab_matrix1,reduced_stab_matrix2):

    return complex(0.,0.)