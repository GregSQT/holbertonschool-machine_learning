#!/usr/bin/python3

"""
Exercice : 
You have n number of locked boxes in front of you.
Each box is numbered sequentially from 0 to n - 1 and each box may contain keys to the other boxes.
Write a method that determines if all the boxes can be opened.
"""


def canUnlockAll(boxes):
    """
    This function wil check if all the boxes can be opened using
    a key found in another box
    """
    if type(boxes) is not list:
        return False
    elif (len(boxes)) == 0:
        return False
    for k in range(1, len(boxes) - 1):
        # We loop on each key number
        boxes_openable = False
        for index in range(len(boxes)):
            # We loop on each box to check if it contains the key (k)
            boxes_openable = k in boxes[index] and k != index
            if boxes_openable:
                # If we have found the key
                break
        if boxes_openable is False:
            # If no key has benn found
            return boxes_openable
    return True
