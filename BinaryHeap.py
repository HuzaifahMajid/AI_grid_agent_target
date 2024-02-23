import random

# recursive minHEAP heap
def minHEAP(list_of_nums, x):
    # This function is for maintain the min-heap property of a binary heap represeneted as a list_of_nums
    #recursivly adjusts the position of the elemnet at index x in the heap to make sure subtree rooted at x satisfies min-heap property
    size = len(list_of_nums)
    left = x*2 +1
    right = x*2 + 2
    sorted = x
    
    if left < size -1 and list_of_nums[left] < list_of_nums[x]:
        sorted = left
    if right < size -1 and list_of_nums[right] < list_of_nums[sorted]:
        sorted = right
    if sorted != x:
        list_of_nums[sorted], list_of_nums[x] = list_of_nums[x], list_of_nums[sorted]
        minHEAP(list_of_nums,sorted)

def pop(list_of_nums,dict_of_cells):
    # pop a random element from a heap represented as a list_of_nums, update the heap to maintain the min-heap property, and return the index of the popped element. 
    #It also takes care of updating the associated dict_of_cellsionary and removing entries if necessary. 
    #The randomness in selecting the element to pop ensures that the heap behaves like a priority queue.
    first = list_of_nums[0]
    cell_num = dict_of_cells[first].pop(random.randrange(len(dict_of_cells[first])))
    if len(dict_of_cells[first]) == 0:
        del dict_of_cells[first]
        list_of_nums[0] = list_of_nums[-1]
        list_of_nums.pop()
        minHEAP(list_of_nums,0)
    return cell_num


# binary heap insert
def add(list_of_nums, dict_of_cells, f_Diff, cell_num):

# This function inserts a value (f_Diff) into a binary heap (list_of_nums) and updates a dict_of_cellsionary (dict_of_cells) based on whether the value is already present in the heap or not. 
# The heap is maintained by performing a heap insertion operation after adding a new element to the heap.
# The dict_of_cellsionary is used to keep track of the indices of values in the heap for efficient lookup.
    
    if f_Diff in list_of_nums: 
        dict_of_cells[f_Diff].append(cell_num)
    else:
        dict_of_cells[f_Diff] = [cell_num]
        x = len(list_of_nums)
        list_of_nums.append(None)
        while x >0 and f_Diff < list_of_nums[(x-1)//2]:
            list_of_nums[x] = list_of_nums[(x-1)//2]
            x = (x-1)//2
        list_of_nums[x] = f_Diff