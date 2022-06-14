README for data set Tox21_MMP_training

=== Usage ===

This folder contains the following comma separated text files 
(replace DS by the name of the dataset):

n = total number of nodes
m = total number of edges
N = number of graphs

(1) 	DS_A.txt (m lines) 
	sparse (block diagonal) adjacency matrix for all graphs,
	each line corresponds to (row, col) resp. (node_id, node_id)

(2) 	DS_graph_indicator.txt (n lines)
	column vector of graph identifiers for all nodes of all graphs,
	the value in the i-th line is the graph_id of the node with node_id i

(3) 	DS_graph_labels.txt (N lines) 
	class labels for all graphs in the dataset,
	the value in the i-th line is the class label of the graph with graph_id i

(4) 	DS_node_labels.txt (n lines)
	column vector of node labels,
	the value in the i-th line corresponds to the node with node_id i

There are OPTIONAL files if the respective information is available:

(5) 	DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt)
	labels for the edges in DS_A_sparse.txt 

(6) 	DS_edge_attributes.txt (m lines; same size as DS_A.txt)
	attributes for the edges in DS_A.txt 

(7) 	DS_node_attributes.txt (n lines) 
	matrix of node attributes,
	the comma seperated values in the i-th line is the attribute vector of the node with node_id i

(8) 	DS_graph_attributes.txt (N lines) 
	regression values for all graphs in the dataset,
	the value in the i-th line is the attribute of the graph with graph_id i

=== Node Label Conversion === 


Node labels were converted to integer values using this map:

Component 0:
	0	O
	1	C
	2	N
	3	F
	4	Cl
	5	S
	6	Br
	7	Si
	8	Na
	9	I
	10	Hg
	11	B
	12	K
	13	P
	14	Au
	15	Cr
	16	Sn
	17	Ca
	18	Cd
	19	Zn
	20	V
	21	As
	22	Li
	23	Cu
	24	Co
	25	Ag
	26	Se
	27	Pt
	28	Al
	29	Bi
	30	Sb
	31	Ba
	32	Fe
	33	H
	34	Ti
	35	Tl
	36	Sr
	37	In
	38	Dy
	39	Ni
	40	Be
	41	Mg
	42	Nd
	43	Pd
	44	Mn
	45	Zr
	46	Pb
	47	Yb
	48	Mo
	49	Ge
	50	Ru
	51	Eu
	52	Sc
	53	Gd



Edge labels were converted to integer values using this map:

Component 0:
	0	-
	1	=
	2	:
	3	#


=== References ===

Tox21 Data Challenge 2014, https://tripod.nih.gov/tox21/challenge/data.jsp

=== Important ===
Not all graphs are included in all Tox21-datasets. "inconclusive/not tested" results were ignored.


