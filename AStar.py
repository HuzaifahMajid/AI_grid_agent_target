import numpy as np
import random
from matplotlib.colors import ListedColormap
import matplotlib.colors as pltColors
import matplotlib.pyplot as plt
import matplotlib 
import BinaryHeap as bh
import time 
import sys
from mazeGen import generateMazes

def DeadEND(y,x,visited) :     
	for i in range(-1,1):   #i in -1,0,1
		for j in range(-1,1):#i in -1,0,1
			if(not(i==0 and j==0)):#as if i==0 and j==0 then we are in teh same cell 	 
				if(x+i>=0 and x+i< num_of_cols ):
					if( y+j >=0 and y+j < num_of_rows):
						if((x+i,y+j) not in visited):
							return False,y+j,x+i #There's an unvisited neighbour 
	return True,-1,-1                            #There's no unvisited neighbour
   
def RowCheck(y):
	if(y>=0 and y<num_of_rows):
		return True  
	return False  

def ColCheck(x):
	if(x>=0 and x<num_of_cols):
		return True  
	return False  
	
def generateMazes(num_of_mazes,num_of_rows,num_of_cols):

	#Generate 50 mazes and initially set all of the cells as unvisited
	maze  = np.zeros((num_of_mazes,num_of_rows,num_of_cols))

	for current_maze in range(0,num_of_mazes) :
		print("Generate Maze : " + str(current_maze+1)) 
		visited = set()  # Set for visitied nodes 
		stack   = []     # Stack is empty at first 

		#random cell starting point	
		row_index = random.randint(0,num_of_rows-1)
		col_index = random.randint(0,num_of_cols-1)
		#mark  as visitied 	
		print("- Start -\n")
		print("Loc["+str(row_index)+"],["+str(col_index)+"] = 1")
		visited.add((row_index , col_index))       #Visited 
		maze [current_maze , row_index , col_index] = 1 #Unblocked 
		
		#Select random neighbouring cell to visit that has not yet been visited. 
		print("\n\- DFS -\n")
		while(len(visited) < num_of_cols*num_of_rows): #Repeat till visit all cells 
		
			crnt_row_index = row_index+random.randint(-1,1)#neighbor
			crnt_col_index = col_index+random.randint(-1,1)#neighbor
			i=0 
			isDead=False 
			while ((not RowCheck(crnt_row_index)) or (not ColCheck(crnt_col_index) )or ((crnt_row_index,crnt_col_index) in visited) ):
				crnt_row_index = row_index+random.randint(-1,1)
				crnt_col_index = col_index+random.randint(-1,1)
				i = i+1
				if(i==8):
					#Reached dead end 
					print("dead end boss")
					isDead = True
					break
			if(not isDead):
				visited.add((crnt_row_index , crnt_col_index)) 
			
			probability  = random.uniform(0, 1)

			if( probability < 0.3 and not isDead) : 
				#  30% probability mark as blocked. 
				maze [current_maze , crnt_row_index , crnt_col_index] = 0 #Leave  block  
				print("Loc["+str(crnt_row_index)+"],["+str(crnt_col_index)+"] = 0")				
				#to start get  neighbors of this cell next time 
				row_index = crnt_row_index
				col_index = crnt_col_index
			else : 
				if(not isDead):
					# With 70% mark it as unblocked and add to stack.
					maze [current_maze , crnt_row_index , crnt_col_index] = 1 #Unblocked 
					print("Loc["+str(crnt_row_index)+"],["+str(crnt_col_index)+"] = 1")				
					stack.append((crnt_row_index,crnt_col_index))
					isDead,explore_row , explore_col = DeadEND(row_index,col_index,visited)
				if(isDead == True):
					#backtrack to parent nodes on the search tree until it reaches a cell with an unvisited neighbour
					while(len(stack)>0):
						parent_row,parent_col = stack.pop() 
						isDead,explore_row , explore_col = DeadEND(parent_row,parent_col,visited)
						if(isDead == False):
							break 
					if(len(stack)>0):
						visited.add((explore_row,explore_col))
						row_index = explore_row
						col_index = explore_col
					else :
						#Repeat from a point not vistited
						row_index = random.randint(0,num_of_rows-1)
						col_index = random.randint(0,num_of_cols-1)
						if(len(visited)< num_of_cols*num_of_rows):
							while ( (not RowCheck(row_index)) or (not ColCheck(col_index)) or ((row_index,col_index) in visited) ):
								row_index = random.randint(0,num_of_rows-1)
								col_index = random.randint(0,num_of_cols-1)
								print(str(row_index)+","+str(col_index))
						#mark it as visitied 	
						visited.add((row_index , col_index))       #Visited 					
				else : #No dead Node 
					visited.add((explore_row,explore_col))
					row_index = explore_row
					col_index = explore_col

		print("maze succesfully printed")	
	return maze
	
	#plot line
def plt_line(A,col):
    x, y = zip(*A)
    temp, = plt.plot(y,x,color=col)
    print("Line type ")
    print(temp.__class__.__name__)
    return temp
	

# This method is to get Manhattan Distances for A*
def DistanceDiff(curr, goal):
    return abs(curr[0] - goal[0]) + abs(curr[1] - goal[1])

# This method will find the path for Forward A*
def getForwardPath(dict, curr, goal):
    temp = [goal]
    next = dict[goal]
    while next != curr:
        temp.insert(0,next)
        next = dict[next]
    temp.insert(0,next)
    return temp

# This method will find the path for the Backward A*
def getBackwardPath(dict, curr, goal):
    temp = [curr]
    next = dict[curr]
    while next != goal:
        temp.append(next)
        next = dict[next]
    temp.append(next)
    return temp

# This method will find the neighbors of the current cell
def nextNeighbor (cellNumber, length):
    neighborList = [(cellNumber[0],cellNumber[1]+1),(cellNumber[0],cellNumber[1]-1),(cellNumber[0]+1,cellNumber[1]),(cellNumber[0]-1,cellNumber[1])]
    temp = []
    for x in neighborList:
        if -1 < x[0] < length and -1 < x[0] < length:
            temp.append(x)
    return temp

# update the neighbors to empty and blocked
def update_status(map,length,curr,bList):
    goodCells = nextNeighbor(curr,length)
    for x in goodCells:
        print(x)
        print(map[2][x])
		
        if x in bList:
            map[2][x] = 2 # It's blocked
        else:
            map[2][x] = 1 # It's empty


# display intial map

def dispMap(length, bList = None, pList = None, map = None, old = None):
    if not(map is None):
        d = map[2]
        f, a = plt.subplots()
        print(d)
        a.imshow(d)
    else:
        d = np.zeros((length, length))
        if not (bList is None):
            x = bList
        f, a = plt.subplots()
        a.imshow(d)
    print(a)
    a.grid(which= 'minor', color='black', linestyle='-', linewidth=1)
    
    if pList != None:
        plt_line(pList, 'red')
    if old != None:
        plt_line(old, 'green')
    plt.show()


def color_maze(length, bList = None, pList = None, map = None, old = None): 
	#add the colors to the maze 
    d = map[2]
    f,a = plt.subplots()
    colorMap = matplotlib.colors.ListedColormap(['purple','white','black'])
    boundaries = [-0.5,0.5,1.5,2.5]
    n = matplotlib.colors.BoundaryNorm(boundaries, colorMap.N)
    image = a.imshow(d,interpolation='nearest', origin='lower')
    a.set_xticks(np.arange(-.5, length, 1), minor=True)
    a.set_yticks(np.arange(-.5, length, 1), minor=True)
    a.grid(which= 'minor', color='orange', linestyle='-', linewidth=1)
    oLine = plt_line(old,'yellow')
    pLine = plt_line(pList, 'red')
    plt.pause(1)
    return image, f, oLine, pLine

def update_graph(image, f, oLine, pLine, map, aList, bList):
	# This method will add the path line to the maze 
    image.set_data(map)
    x,y = zip(*aList)
    oLine.set_xdata(y)
    oLine.set_ydata(x)
    x,y = zip(*bList)
    pLine.set_xdata(y)
    pLine.set_ydata(x)
    plt.pause(1)
    f.canvas.draw()

def adaptedAStar(mLength,bList, start, goal, show = False):
	# Here we implement Adaptive A* which will take the length, start, and goal as inputs
	maximumG = mLength * mLength
	currCell = start 
	gCell = goal
	map_array = np.zeros((5,mLength, mLength))
	map_array[3] = np.zeros((mLength,mLength), dtype = bool)
	map_array[4] = np.zeros((mLength,mLength), dtype = bool)
	map_array[2][currCell] = 1
	maxG = 0 
	totalSteps = 0 
	totalExpense = 0 
	currTrack = [start]
	update_status(map_array,mLength,currCell,bList)
	while currCell != gCell:
		totalSteps += 1 
		map_array[1][currCell] = 0
		map_array[0][currCell] = totalSteps
		map_array[1][gCell] = np.inf
		map_array[0][gCell] = totalSteps
		oList = []
		oDict = dict()
		fDict = dict()
		map_array[4] = np.zeros((mLength,mLength), dtype = bool)
		bh.add(oList, oDict, DistanceDiff(currCell,gCell), currCell)
		while oList and map_array[1][gCell] > oList[0]:
			coloredCell = bh.pop(oList, oDict)
			map_array[4][coloredCell] = True
			totalExpense += 1
			for newCell in nextNeighbor(coloredCell,mLength):
				if map_array[2][newCell] != 2 :
					if map_array[3][newCell]:
						newHeuris = maxG - map_array[1][newCell]
					else:
						newHeuris = DistanceDiff(newCell,gCell)
					if map_array[0][newCell] < totalSteps:
						map_array[1][newCell] = np.inf
						map_array[0][newCell] = totalSteps
					if map_array[1][newCell] > map_array[1][coloredCell] + 1:
						fDict[newCell] = coloredCell
						newG = map_array[1][coloredCell] + 1
						bh.add(oList, oDict, (maximumG*( newG +  newHeuris) -  newG) , newCell )
						map_array[1][newCell] = newG
		map_array[3] = map_array[4]
		maxG = map_array[1][gCell]
		if not oList:
			return None, None
		currPath = getForwardPath(fDict,currCell,gCell)
		if show:
			if currCell == start:
				plt.ion()
				im, fig,oLine,pLine = color_maze(mLength, map = map_array, pList = currPath , old = currTrack)
				plt.show()
			else:
				update_graph(im, fig, oLine,pLine, map_array[2],  currTrack ,currPath)
		for cell in currPath:
			if cell == currCell:
				continue
			else:
				if map_array[2][cell] != 2 :
					currTrack.append(cell)
					currCell = cell
					update_status( map_array,mLength,currCell,bList)
				else:
					break 
	if show:
		update_graph(im, fig, oLine,pLine, map_array[2],  currTrack ,currPath)
	return currTrack,totalExpense

def backwardAStarTie(mLength,bList, start, goal, show = False):
	# Here we implement Backward A* which will take the length, start, and goal as inputs
	totalExpense = 0 
	maxG = mLength*mLength
	currCell = start 
	gCell = goal
	map_array = np.zeros((4,mLength, mLength))
	map_array[2][currCell] = 1
	totalSteps = 0 
	currTrack = [start]
	update_status(map_array,mLength,currCell,bList)
	while currCell != gCell:
		map_array[3] = np.zeros((mLength,mLength))
		totalSteps += 1 
		map_array[1][currCell] = np.inf
		map_array[0][currCell] = totalSteps
		map_array[1][gCell] = 0
		map_array[0][gCell] = totalSteps
		oList = []
		oDict = dict()
		fDict = dict()
		bh.add(oList, oDict, DistanceDiff(currCell,gCell), gCell)
		while oList and map_array[1][currCell] > oList[0]:
			coloredCell = bh.pop(oList, oDict)
			map_array[3][coloredCell] = 1
			totalExpense += 1
			for newCell in nextNeighbor(coloredCell,mLength):
				if map_array[2][newCell] != 2 :
					if map_array[0][newCell] < totalSteps:
						map_array[1][newCell] = np.inf
						map_array[0][newCell] = totalSteps
					if map_array[1][newCell] > map_array[1][coloredCell] + 1:
						map_array[1][newCell] = map_array[1][coloredCell] + 1
						fDict[newCell] = coloredCell
						bh.add(oList, oDict, maxG*(map_array[1][newCell] + DistanceDiff(newCell,currCell)) - map_array[1][newCell] , newCell )
		if not oList:
			return None, None
		
		if show:
			currPath = getBackwardPath(fDict,currCell,gCell)
			if currCell == start:
				plt.ion()
				im, fig,oLine,pLine = color_maze(mLength, map = map_array, pList = currPath , old = currTrack)
				plt.show()
			else:
				update_graph(im, fig, oLine,pLine, map_array[2],  currTrack ,currPath)
		while currCell != gCell:
			cell = fDict[currCell]
			if map_array[2][cell] != 2 :
				currTrack.append(cell)
				currCell = cell
				update_status( map_array,mLength,currCell,bList)
			else:
				break
	if show:
		update_graph(im, fig, oLine,pLine, map_array[2],  currTrack ,currPath) 
	return currTrack, totalExpense

def forwardAStarTie(mLength,bList, start, goal, show = False):
	# Here we implement Forward A* which will take the length, start, and goal as inputs
	totalExpense = 0
	maxG = mLength* mLength
	currCell = start 
	gCell = goal
	map_array = np.zeros((4,mLength, mLength))
	map_array[2][currCell] = 1
	totalSteps = 0 
	currTrack = [start]
	update_status(map_array,mLength,currCell,bList)
	while currCell != gCell:
		totalSteps += 1 
		map_array[1][currCell] = 0
		map_array[0][currCell] = totalSteps
		map_array[1][gCell] = np.inf
		map_array[0][gCell] = totalSteps
		oList = []
		oDict = dict()
		fDict = dict()
		bh.add(oList, oDict, DistanceDiff(currCell,gCell), currCell)
		while oList and map_array[1][gCell] > oList[0]:
			coloredCell = bh.pop(oList, oDict)
			map_array[3][coloredCell] = 1
			totalExpense += 1
			for newCell in nextNeighbor(coloredCell,mLength):
				if  map_array[2][newCell] != 2 :
					if map_array[0][newCell] < totalSteps:
						map_array[1][newCell] = np.inf
						map_array[0][newCell] = totalSteps
					if map_array[1][newCell] > map_array[1][coloredCell] + 1:
						map_array[1][newCell] = map_array[1][coloredCell] + 1
						fDict[newCell] = coloredCell
						bh.add (oList, oDict, ( maxG*(map_array[1][newCell] + DistanceDiff(newCell,gCell)) - map_array[1][newCell] )  , newCell ) 
		if not oList:
			return None, None
		currPath = getForwardPath(fDict,currCell,gCell)
		if show:
			if currCell == start:
				plt.ion()
				im, fig,oLine,pLine = color_maze(mLength, map = map_array, pList = currPath , old = currTrack)
				plt.show()
			else:
				update_graph(im, fig, oLine,pLine, map_array[2],  currTrack ,currPath)
		for cell in currPath:
			if cell == currCell:
				continue
			else:
				if map_array[2][cell] != 2 :
					currTrack.append(cell)
					currCell = cell
					update_status( map_array,mLength,currCell,bList)
				else:
					break 
	if show:
		update_graph(im, fig, oLine,pLine, map_array[2],  currTrack ,currPath)
	return currTrack, totalExpense

def repeatedForwardAStar(mLength,bList, start, goal, show = False):
	# Here we implement Repeated Forward A* which will take the length, start, and goal as inputs
	totalExpense = 0
	maxG = mLength* mLength
	currCell = start 
	gCell = goal
	map_array = np.zeros((4,mLength, mLength))
	map_array[2][currCell] = 1
	totalSteps = 0 
	currTrack = [start]
	update_status(map_array,mLength,currCell,bList)
	while currCell != gCell:
		totalSteps += 1 
		map_array[1][currCell] = 0
		map_array[0][currCell] = totalSteps
		map_array[1][gCell] = np.inf
		map_array[0][gCell] = totalSteps
		oList = []
		oDict = dict()
		fDict = dict()
		bh.add(oList, oDict, DistanceDiff(currCell,gCell), currCell)
		while oList and map_array[1][gCell] > oList[0]:
			coloredCell = bh.pop(oList, oDict)
			totalExpense += 1
			for newCell in nextNeighbor(coloredCell,mLength):
				if  map_array[2][newCell] != 2 :
					if map_array[0][newCell] < totalSteps:
						map_array[1][newCell] = np.inf
						map_array[0][newCell] = totalSteps
					if map_array[1][newCell] > map_array[1][coloredCell] + 1:
						map_array[1][newCell] = map_array[1][coloredCell] + 1
						fDict[newCell] = coloredCell
						bh.add (oList, oDict, ( maxG*(map_array[1][newCell] + DistanceDiff(newCell,gCell)) + map_array[1][newCell] )  , newCell ) 
		if not oList:
			return None, None
		currPath = getForwardPath(fDict,currCell,gCell)
		if show:
			if currCell == start:
				plt.ion()
				im, fig,oLine,pLine = color_maze(mLength, map = map_array, pList = currPath , old = currTrack)
				plt.show()
			else:
				update_graph(im, fig, oLine,pLine, map_array[2],  currTrack ,currPath)
		for cell in currPath:
			if cell == currCell:
				continue
			else:
				if map_array[2][cell] != 2 :
					currTrack.append(cell)
					currCell = cell
					update_status( map_array,mLength,currCell,bList)
				else:
					break 
	if show:
		update_graph(im, fig, oLine,pLine, map_array[2],  currTrack ,currPath)
	return currTrack, totalExpense

def repeatedForwardAStarB(mLength,bList, start, goal, show = False):
	# Here we implement Repeated Forward A* which will take the length, start, and goal as inputs
	totalExpense = 0
	currCell = start 
	gCell = goal
	map_array = np.zeros((3,mLength, mLength))
	map_array[2][currCell] = 1
	totalSteps = 0 
	currTrack = [start]
	update_status(map_array,mLength,currCell,bList)
	while currCell != gCell:
		totalSteps += 1 
		map_array[1][currCell] = 0
		map_array[0][currCell] = totalSteps
		map_array[1][gCell] = np.inf
		map_array[0][gCell] = totalSteps
		oList = []
		oDict = dict()
		fDict = dict()
		bh.add(oList, oDict, DistanceDiff(currCell,gCell), currCell)
		while len(oList)>0 and map_array[1][gCell] > oList[0]:
			coloredCell = bh.pop(oList, oDict)
			totalExpense +=1
			for newCell in nextNeighbor(coloredCell,mLength):
				if  map_array[2][newCell] != 2 :
					if map_array[0][newCell] < totalSteps:
						map_array[1][newCell] = np.inf
						map_array[0][newCell] = totalSteps
					if map_array[1][newCell] > map_array[1][coloredCell] + 1:
						map_array[1][newCell] = map_array[1][coloredCell] + 1
						fDict[newCell] = coloredCell
						bh.add(oList, oDict, (map_array[1][newCell] + DistanceDiff(newCell,gCell)) , newCell )
		if not oList:
			return None, None
		currPath = getForwardPath(fDict,currCell,gCell)
		if show:
			
			if currCell == start:
				plt.ion()
				im, fig,oLine,pLine = color_maze(mLength, map = map_array, pList = currPath , old = currTrack)
				plt.show()
			else:
				update_graph(im, fig, oLine,pLine, map_array[2],  currTrack ,currPath)
		for cell in currPath:
			if cell == currCell:
				continue
			else:
				if map_array[2][cell] != 2 :
					currTrack.append(cell)
					currCell = cell
					update_status( map_array,mLength,currCell,bList)
				else:
					break 
	if show:
		update_graph(im, fig, oLine,pLine, map_array[2],  currTrack ,currPath)
	return currTrack,totalExpense

def repeatedBackwardAStar(mLength,bList, start, goal, show = False):
	# Here we implement Repeated Backward A* which will take the length, start, and goal as inputs
	totalExpense = 0
	currCell = start 
	gCell = goal
	map_array = np.zeros((3,mLength, mLength))
	map_array[2][currCell] = 1
	totalSteps = 0 
	currTrack = [start]
	update_status(map_array,mLength,currCell,bList)
	while currCell != gCell:
		totalSteps += 1 
		map_array[1][currCell] = np.inf
		print("map_array")
		print(map_array [1,1,2])
		print("currCell")
		print( currCell )
		
		
		map_array[0][currCell] = totalSteps
		map_array[1][gCell] = 0
		map_array[0][gCell] = totalSteps
		oList = []
		oDict = dict()
		fDict = dict()
		bh.add(oList, oDict, DistanceDiff(currCell,gCell), gCell)
		print("olist")
		print(oList)
		while oList :
			print(map_array[1][currCell])
			
			if(map_array[1][currCell] <= oList[0]):
				break 
			totalExpense += 1
			coloredCell = bh.pop(oList, oDict)
			for newCell in nextNeighbor(coloredCell,mLength):
				if map_array[2][newCell] != 2 :
					if map_array[0][newCell] < totalSteps:
						map_array[1][newCell] = np.inf
						map_array[0][newCell] = totalSteps
					if map_array[1][newCell] > map_array[1][coloredCell] + 1:
						map_array[1][newCell] = map_array[1][coloredCell] + 1
						fDict[newCell] = coloredCell
						bh.add(oList, oDict,map_array[1][newCell] + DistanceDiff(newCell,currCell), newCell )
		if not oList:
			return None, None
		
		if show:
			currPath = getBackwardPath(fDict,currCell,gCell)
			if currCell == start:
				plt.ion()
				im, fig,oLine,pLine = color_maze(mLength, map = map_array, pList = currPath , old = currTrack)
				plt.show()
			else:
				update_graph(im, fig, oLine,pLine, map_array[2],  currTrack ,currPath)
		while currCell != gCell:
			cell = fDict[currCell]
			if map_array[2][cell] != 2 :
				currTrack.append(cell)
				currCell = cell
				update_status( map_array,mLength,currCell,bList)
			else:
				break 
	if show:
		update_graph(im, fig, oLine,pLine, map_array[2],  currTrack ,currPath)
	return currTrack, totalExpense




if __name__ == '__main__':


	num_of_rows    = 101
	num_of_cols    = 101
	num_of_mazes = 50

	
	num_of_rows    = 5
	num_of_cols    = 5
	num_of_mazes = 3
	mazes2 = generateMazes(num_of_mazes,num_of_rows,num_of_cols)

	mLength = 5 
	bList = [(2,3),(3,4),(3,3),(4,3),(4,4),(5,4)] 
	start = (1,1) 
	goal  = (3,2) 
	
	#np.savetxt('maze '+str(current_maze)+'.txt',mazes[current_maze].astype(int) ,fmt='%i', delimiter=",") 

	length= 5             
	bList = [(2,3),(3,4),(3,3),(4,3),(4,4),(5,4)]  # Example given in assignment
	pList = [(3,1),(3,4)] 
	map   = mazes2      
	old   = None          
	dispMap(list, bList  , pList  , map  , old  ) 
	
	print("Repeated Forward A*")
	repeatedForwardAStarB(mLength,bList, start, goal, True) 


	print("Repeated Backward A*")
	repeatedBackwardAStar(mLength,bList, start, goal, True) 

	
	print("Repeated Forward A* with smaller g-values")
	repeatedForwardAStar(mLength,bList, start, goal, True) 

	print("Adapted A*")
	adaptedAStar(mLength,bList, start, goal, True) 

	print("Repeated Forward A* with large g-values")
	forwardAStarTie(mLength,bList, start, goal, True) 
	print("____________________________________")

	print("Repeated Backward A* with large g-values")		
	backwardAStarTie(mLength,bList, start, goal, True) 
	
	