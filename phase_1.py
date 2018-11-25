"""

    #Last Edit : 24:11:18 19:35:21
    mini log:
            #added constraints for addition and deletion
            #imporved bug for bool has no attribute 

"""

import copy #for deepcopy
import math #for maths intensive operations
import time #for execution time computation
#from numba import jit

def deep_copy(adj2):    #auxilary module for copying the adjacency list
    var={}
    var=copy.deepcopy(adj2)
    return var

# its an ordered entry with {Node, Type, CT, Delay, Weight of Edge}
adj={} #dictionary for adjacency list
node_CT={} #stores computation time corresponding to each node

def initialize(raw_file):   #first function for parsing the input and getting all nodes
    data=open(raw_file,'r')
    if not data:
        print "File doesn't exist"
    adj['GInput']=[]   #in adj list dic we created two lists
    adj['GOutput']=[]
    node_CT['GInput']=0   #same in node computation time dictionary
    node_CT['GOutput']=0
    for line in data :
        arr=line.split()
        a=line[line.find("(")+1:line.find(")")]
        if '#' not in line and line!='\n' and 'INPUT' not in line and 'OUTPUT' not in line:
            adj[arr[0]]=[] #we store all nodes provided on lhs
        if 'INPUT' in line or 'OUTPUT' in line :
            adj[a]=[] #and we mark input or output nodes

is_delete = False #for tracking if any node has come for deletion
is_add = False #for tracking addition  
Input=[]    #list for IP
Output=[]
Add = []    #for addition of extra registers
Del_1 = []  #for deletion of extra registers
Del_2 = []
DFF_list=[] #for storing the nodes associated with registeres
DFF_CT = INPUT_CT = OUTPUT_CT = 0   #computational delays
NOR_CT = NOT_CT = AND_CT = NAND_CT = OR_CT = 1
NodeTypes={}    #the dictionary stores the type of each node

def build_graph(raw_file): #Here all the operations are getting filtered
    data=open(raw_file,'r')
    for line in data :  #here parsing of the input file takes place
        var=[] 
        var2=[]
        arr=line.split()
        if '#' not in line: #we read if its not a comment 
            a=line[line.find("(")+1:line.find(")")]
            a1=a.split(',')
            if 'INPUT' in line: #we find the input nodes
                Input.append(a) #we append it in input list
                node_CT[a]=INPUT_CT
            elif 'OUTPUT' in line:  #we find the output nodes
                Output.append(a)
                node_CT[a]=OUTPUT_CT
            elif 'AND' in line and 'NAND' not in line:
                for i in range(0,len(a1)):
                    m=a1[i].split()
                    node_CT[arr[0]]=AND_CT
                    NodeTypes[arr[0]]='AND'
                    adj[m[0]].append([arr[0],'AND',AND_CT,0,0])
            elif 'OR' in line and 'NOR' not in line:
                for i in range(0,len(a1)):
                    m=a1[i].split()
                    node_CT[arr[0]]=OR_CT
                    NodeTypes[arr[0]]='OR'
                    adj[m[0]].append([arr[0],'OR',OR_CT,0,0])
            elif 'NAND' in line:
                for i in range(0,len(a1)):
                    m=a1[i].split()
                    node_CT[arr[0]]=NAND_CT
                    NodeTypes[arr[0]]='NAND'
                    adj[m[0]].append([arr[0],'NAND',NAND_CT,0,0])
            elif 'NOR' in line :
                for i in range(0,len(a1)):
                    m=a1[i].split()
                    node_CT[arr[0]]=NOR_CT
                    NodeTypes[arr[0]]='NOR'
                    adj[m[0]].append([arr[0],'NOR',NOR_CT,0,0])
            elif 'NOT' in line:
                for i in range(0,len(a1)):
                    m=a1[i].split()
                    node_CT[arr[0]]=NOT_CT
                    NodeTypes[arr[0]]='NOT'
                    adj[m[0]].append([arr[0],'NOT',NOT_CT,0,0])
            elif 'DFF' in line :
                for i in range(0,len(a1)):
                    m=a1[i].split()
                    node_CT[arr[0]]=DFF_CT
                    DFF_list.append(arr[0])
                    adj[m[0]].append([arr[0],'DFF',DFF_CT,0,0])

def create_retiming_model(adj): #Here we populate the adj list with edge weights wrt IP & OP
    for i in range(0,len(Input)):
        adj['GInput'].append([Input[i],'INPUT',0,0,0])

    for i in range(0,len(Output)):
        adj[Output[i]].append(['GOutput','OUTPUT',0,0,0])

    var = copy.deepcopy(adj)
    return var

def weight_builder_DFS(adj, entry, i, p):   #Now we populate weights for rest of the nodes
    varNode=adj[entry][i][0]
    if(adj[varNode][0][1]=='DFF'): #if its DFF we keep moving forward
        p = p+1 #since we need to create delay
        return weight_builder_DFS(adj,varNode,i,p)
    else :
        adj[varNode][0][4]=adj[varNode][0][4]+1 + p    #we update weights
        return adj[varNode][0]

def weighted_graph_population(adj):
    for entry in adj:   #for every node in our adj list
        for i in range(0, len(adj[entry])):
            #print entry, adj[entry]
            if(adj[entry][i][1]=='DFF'):  #if the node type is DFF
                varNode = weight_builder_DFS(adj,entry,i,0)
                #varNode is the address location of the node
                var =[varNode[0],varNode[1],varNode[2],varNode[3],varNode[4]]
                #storing in the value not the address of the var node
                adj[entry][i]=var

unit_time = 1   #can be changed based on needs
keys=[]
keys2=[]

def repopulate_weighted_graph(adj_reweighted):
    n=0  
    for entry in adj_reweighted:
        if entry not in DFF_list:
            n=n+1   #counting entries which are not DFF
    M = unit_time*n
    for entry in adj_reweighted:
        for i in range(0,len(adj_reweighted[entry])):
            if(node_CT[entry]!=0):
                adj_reweighted[entry][i][4]=(M*adj_reweighted[entry][i][4])-node_CT[entry]

def form_adj_mat(adj):
    mat=[[float('inf') for x in range(len(adj)-len(DFF_list))] for y in range(len(adj)-len(DFF_list))]
    for i in range(len(mat[0])):
        mat[i][i]=0
    for entry in adj :
        if entry not in DFF_list:
            keys.append(entry)
    for entry in adj:
        if entry not in DFF_list:
            l=keys.index(entry)
            for i in range(len(adj[entry])):
                entry2=adj[entry][i][0]
                if entry2 not in DFF_list:
                    m=keys.index(entry2)
                    mat[l][m]=adj[entry][i][4]
    return mat

def floydWarshall_unoptimised(mat):    #space n^3 and time O(n^3)
    n=len(mat[0])
    D=[None for x in range(0,n)]
    D[0]=mat
    for k in range(1,n):
        D[k]=[[None for x in range(0,n)]for y in range(0,n)]
        for i in range(0,n):
            for j in range(0,n):
                D[k][i][j]=min(D[k-1][i][j],D[k-1][i][k]+D[k-1][k][j])
    return D[n-1]

def floydWarshall(mat): #space O(n^2) and time n^2logn
    n=len(mat[0])
    for k in range(1,n):
        for i in range(0,n/2+1):
            for j in range(0,n):
                if(mat[i][k] != float('inf') and mat[k][j] != float('inf') and mat[i][j] > mat[i][k] + mat[k][j]):
                    mat[j][i] = mat[i][j] = mat[i][k] + mat[k][j]
    return mat

def populate_w_matrix(mat): #minimum registers in u,v
    n=len(mat[0])
    M=unit_time*n
    W_mat=[[0 for x in range(0,n)]for y in range(0,n)]
    for i in range(0,n):
        for j in range(0,n):
            W_mat[i][j] = math.ceil(mat[i][j]/M)
    return W_mat

def populate_d_matrix(mat,W_mat,adj): #max computational delay with weight(u, v)
    n=len(mat[0])
    M=unit_time*n
    D_mat=[[0 for x in range(0,n)]for y in range(0,n)]
    for i in range(0,n):
        for j in range(0,n):
            if(i==j):
                    D_mat[i][j]=node_CT[keys[i]]
            else :
                    D_mat[i][j]=W_mat[i][j]*M-mat[i][j]+node_CT[keys[j]]
    return D_mat

def constraints(W_mat,D_mat,mat,adjList,keys,c):    #for handlig the constraints for retiming 
    inequalityDict={}
    inequalityDict_F = {}

    for i in range(0,len(keys)):
        inequalityDict[keys[i]]=[]

    """ 
        Here we resolve 4 kind of constraints
        Now we can have either deletion or additon
        So we check for any one or none and then 
        We add the constraints
    """

    #adding constraints for addition and deletion    
    if(is_add):
        #print "In addition "
        for i in range(0,len(mat[0])):
            for j in range(0,len(mat[0])):
                u=keys[i]
                v=keys[j]
                var = 0
                for k in range(0, len(Add)):
                    if( Add[k] == v ):  #we check for the node after which it has to be added 
                        var = 0
                for k in range(0,len(adjList[keys[i]])):
                    if(adjList[keys[i]][k][0] == keys[j]):
                        inequalityDict[v].append( [u, adjList[keys[i]][k][4] + var  ] )
    
    elif(is_delete):
        #print "In deletion "
        for i in range(0,len(mat[0])):
            for j in range(0,len(mat[0])):
                u=keys[i]
                v=keys[j]
                var = 0
                for k in range(0, len(Del_1)):
                    #here we try to figure out the edge for adding constraints 
                    if( (Del_1[k] == v and Del_2[k] == u) or (Del_1[k] == v or Del_2[k] == u ) ):
                        var = 1
                for k in range(0,len(adjList[keys[i]])):
                    if(adjList[keys[i]][k][0]==keys[j]):
                        inequalityDict[v].append( [u, adjList[keys[i]][k][4] + var ] )
    
    else:
        #print "In no constraint "
        #this is the normal retiming when we have no addition or deletion 
        for i in range(0,len(mat[0])):
            for j in range(0,len(mat[0])):
                u=keys[i]
                v=keys[j]
                for k in range(0,len(adjList[keys[i]])):
                    if(adjList[keys[i]][k][0]==keys[j]):
                        inequalityDict[v].append( [u, adjList[keys[i]][k][4] ] )

    #now we go for our conventional constraints 

    for i in range(0,len(mat[0])):  #feasibility constraint
        for j in range(0,len(mat[0])):
            u=keys[i]
            v=keys[j]
            for k in range(0,len(adjList[keys[i]])):
                if(adjList[keys[i]][k][0] < W_mat[i][j] ):
                    inequalityDict_F[v].append( [u, adjList[keys[i]][k][4] ] )

    for i in range(0,len(mat[0])):  #critical path constraint 
        for j in range(0,len(mat[0])):
            u=keys[i]
            v=keys[j]
            if(D_mat[i][j] > c):    #critical path constraint is being checked
                flag=0
                weight=W_mat[i][j]-1
                if(inequalityDict[v]==[]):
                    inequalityDict[v].append([  u,weight ])
                else:
                    for k in range(0,len(inequalityDict[v])):
                        if(inequalityDict[v][k][0]==u):
                            flag=1
                            if(inequalityDict[v][k][1]>weight):

                                inequalityDict[v][k][1]=weight
                    if(flag==0):
                        inequalityDict[v].append([ u, weight])

    for key, item in inequalityDict.items():
        if item == []:
            del inequalityDict[key]

    for entry in inequalityDict:
        if(entry not in keys2):
            keys2.append(entry)
        for j in range(0,len(inequalityDict[entry])):
            if(inequalityDict[entry][j][0] not in keys2 ):
                keys2.append(inequalityDict[entry][j][0])

    mat2=[[ float('inf') for x in range(0,len(keys2)+1)] for y in range(0,len(keys2)+1) ]

    for entry in inequalityDict:
        for k in range(0,len(inequalityDict[entry])):
            i=keys2.index(entry)
            j=keys2.index(inequalityDict[entry][k][0])
            mat2[i][j]=inequalityDict[entry][k][1]

    #the last element of the matrix is used as a source that is connected to every node by weight 1
    for j in range(len(mat2[0])):
        mat2[len(keys2)][j]=0   #this will act as an source for bellman ford algorithm 
    return  mat2

def bellmanFord(mat,s): #our single source shortest path algorithm of choice 
    d=[float('inf') for x in range(len(mat[0]))]
    d[len(mat[0])-1]=0

    for i in range(0,len(mat[0])):
        for j in range(0,len(mat[0])):
            if(mat[i][j]!=float('inf') and d[j]>d[i]+mat[i][j]):
                d[j]=d[i]+mat[i][j]   #relasing the edges by bellman ford no need to find parent component

    for i in range(0,len(mat[0])):
        for j in range(0,len(mat[0])):
            if(mat[i][j]!=float('inf') and d[j]>d[i]+mat[i][j]):
                return False

    return d

def bellmanFord_unpotimised(mat,s): #does n/2 more operations per iterations
    d=[float('inf') for x in range(len(mat[0]))]
    d[len(mat[0])-1]=0

    for i in range(0,len(mat[0])):
        for j in range(0,len(mat[0])):
            if(mat[i][j]!=float('inf')):
                if(d[j]>d[i]+mat[i][j]):
                    d[j]=d[i]+mat[i][j]   ####relasing the edges by bellman ford no need to find parent component

    for i in range(0,len(mat[0])):
        for j in range(0,len(mat[0])):
            if(mat[i][j]!=float('inf')):
                if(d[j]>d[i]+mat[i][j]):
                    return False

    return d

def finale_graph(arr,adj):  #we remove the extra nodes added before 
    adjFinal=deep_copy(adj)
    for key, item in adjFinal.items():
        if key not in keys:
            del adjFinal[key]

    for entry in adjFinal:
        for i in range(0,len(adjFinal[entry])):
            adjFinal[entry][i][4]=adjFinal[entry][i][4]+arr[keys2.index(adjFinal[entry][i][0])]-arr[keys2.index(entry)]

    return  adjFinal

def print_to_file(INPUTS,OUTPUTS,dffs,GateList):
    FileP=open('output.txt','w')

    #writing all the INPUTS
    for i in range(0,len(INPUTS)):
        FileP.write('INPUT'+'('+INPUTS[i]+')'+'\n')
    FileP.write('\n')

    #writing all the OUTPUTs
    for i in range(0,len(OUTPUTS)):
        FileP.write('OUTPUT'+'('+OUTPUTS[i]+')'+'\n')
    FileP.write('\n')

    #writting the dffs
    for entry in dffs:
        previous=entry
        for j in range(0,dffs[entry][1]):
            new='DFF' + '_' + entry + '_'+ str(j)
            FileP.write(new + ' = '+'DFF'+'('+previous+')'+'\n')
            previous=new

    FileP.write('\n')
    for entry in GateList:
        var_str=''
        for j in range(1,len(GateList[entry])-1):
            var_str=var_str+GateList[entry][j]+','
        var_str=var_str+GateList[entry][len(GateList[entry])-1]
        FileP.write(entry +' = ' + GateList[entry][0] + '('+var_str+')'+'\n')

def print_back(adjFinal):
    dffs={}
    INPUTS=[]
    OUTPUTS=[]
    GateList={}
    for entry in adjFinal:
        if(entry!='GInput' and entry!='GOutput' and entry not in Input):
            GateList[entry]=[NodeTypes[entry]]


    for i in range(0,len(adjFinal['GInput'])):
        INPUTS.append(adjFinal['GInput'][i][0])
    for i in range(0,len(Output)):
        OUTPUTS.append(Output[i])

    for entry in adjFinal:
        if(entry!='GInput' and entry!='GOutput'  ):
            for i in range(0,len(adjFinal[entry])):
                if(adjFinal[entry][i][4]==0 and adjFinal[entry][i][0]!='GOutput'):
                    head=adjFinal[entry][i][0]
                    GateList[head].append(entry)
                elif(adjFinal[entry][i][4]!=0):
                    head=adjFinal[entry][i][0]
                    GateList[head].append('DFF'+'_'+entry+'_'+str(adjFinal[entry][i][4]-1))
                    dffs[entry]=[head,adjFinal[entry][i][4]]

    print_to_file(INPUTS,OUTPUTS,dffs,GateList)

#   _______________________ Execution flow ___________________________
raw_file = raw_input("Enter the file name : ")
c = int(raw_input('Enter the value of the c : '))
print("Processing " + raw_file + " . . . ")

#Parsing th input file and storing all the nodes and I/P & O/P
initialize(raw_file)

#here we create the adj list corresponding to the list provided
build_graph(raw_file)

"""

                Old logic for addition and deletion 
                ```````````````````````````````````

choice = int(raw_input("To add one or more registers press 1 else 0 : "))

#here we increase the weight of the node to c so that a register gets added in order to break the 
#critical path 
if( choice == 1 ):
    print("Welcome to register adding menu ")
    choice = int(raw_input("Number of registers to be added : "))
    cr = 1
    while(choice != 0):
        choice -= 1
        ad = raw_input("Enter the node number " + str(cr) + " : ")
        for entry in adj:
            if(entry == ad):
                for i in range(0, len(adj[entry])):
                    adj[entry][i][4] = c 
            #for i in range(0, len(adj[entry])):
            #    if(adj[entry][i][0]==ad):
            #        adj[entry][i][4] = c 
        cr += 1
        print "Successfully added a register after " + ad

size = 0

choice = int(raw_input("To delete one or more registers press 1 else 0 : "))

#here we increase the weight of the nodes rechable from the current node by c 
#so that a register gets added in order to break the critical path 
if( choice == 1 ):
    print("Welcome to register deleting menu ")
    print("Please note that registers will only be deleted if they are present ")
    choice = int(raw_input("Number of registers to be deleted : "))
    cr = 1
    while(choice != 0):
        choice -= 1
        ad = raw_input("Enter the node number " + str(cr) + " : ")
        for entry in adj:
            if(entry == ad):
                for i in range(0, len(adj[entry])):
                    adj[entry][i][4] = c 
        cr += 1
        print "Successfully deleted a register after " + ad

#for entry in adj:   #for every node in our adj list
    #if(entry != 'GOutput' and entry != 'GInput'):
    #print entry, adj[entry]
    #size += 1

            New logic has been implemented below
            ````````````````````````````````````
"""

choice = int(raw_input("For deletion press 1 : For addition 2 : "))

if ( choice == 1 ):
    choice = int(raw_input("Enter the number of registers to be deleted "))
    while(choice != 0):
        choice -= 1
        print ( "Enter the edge ")
        e1 = raw_input("Enter the first node : ")
        e2 = raw_input("Enter the second node : ")
        cur = False
        for entry in DFF_list:
            if( e2 == entry ):
                cur = True
                Del_1.append(e1)    #we keep a recod of all the registers to be deleted here 
                Del_2.append(e2)
        if( cur == False ):
            print "Node doesn't exist or registers do not exist "
        if( cur ):
            print "Node found with register, will be deleted : "
        if( is_delete == False ):
            is_delete = cur

elif (choice == 2):
    print("Welcome to register adding menu ")
    choice = int(raw_input("Number of registers to be added : "))
    cr = 1
    while(choice != 0):
        choice -= 1
        ad = raw_input("Enter the node number " + str(cr) + " : ")
        for entry in adj:
            if(entry == ad):
                for i in range(0, len(adj[entry])):
                    adj[entry][i][4] += 1
        Add.append(ad)  #we keep track of all the registers to be added
        is_add = True
        cr += 1
        print "Successfully added a register after " + ad


#print "Total nodes (Leaving input/output nodes) : " + str(size)
start = time.time()

#we store the unweighted adjacency list in an another tuple
adjListInitial = create_retiming_model(adj)
#----------------------------------------------------------------------
#Now data has been processed we create the model required
#print("Creating retiming model | Time stamp : " + str(time.time()-start))
#create retiming model, we go to the previous dependencies of all DFFs and we count the edge weights
weighted_graph_population(adj)

#print("Warshall in pogress | Time stamp : " + str(time.time()-start))
#All pairs shortest path find

adjList=copy.deepcopy(adj)
repopulate_weighted_graph(adj)
mat = form_adj_mat(adj)

#-----------------------------------------------------------------------
#Here floyd warshall is being applied for all pair shortest path
warshall_result=floydWarshall(mat)  #applying optimised warshall

print("Populating D and W matrix | Time stamp : " + str(time.time()-start))

#finding D and W matrix
W_matrix=populate_w_matrix(warshall_result)
D_matrix=populate_d_matrix(warshall_result, W_matrix, adj)

#------------------------------------------------------------------------
#here we change the code depending upon the value of c
#print("Finding Graph inequalities | Time stamp : " + str(time.time()-start))
#finding the graph by inequality

"""
    Here we can have 4 kinds of constraints  
    1) critical path
    2) feasibility 
    3) constraint beacause of addition of registers
    4) constraint because of deletion of registers
    All of them have been taken care in cnstraints function
"""

InequalityMatrix=constraints(W_matrix,D_matrix,mat,adjList,keys,c)

#the last element of the matrix is the source of the bellmanFord
ford_output=bellmanFord(InequalityMatrix,len(InequalityMatrix)-1)   
adjFinal=finale_graph(ford_output,adjList)

#-------------------------------------------------------------------------
#the model has been generated, we just have to write it back to file
#print("Creating output file | Time stamp : " + str(time.time()-start))
print_back(adjFinal)

total = time.time() - start

#for entry in adjList:
#print entry, adjList[entry]

print "Total execution time : " + str(total) 
