# graph2image
A python tool to pack the contour of an image with the nodes of a network graph.

## Example
This social network for the character interactions in the Witcher novels was created by [Milán Janosov](https://github.com/milanjanosov). 
You can check out his work in his article [A Network Map of The Witcher](https://nightingaledvs.com/a-network-map-of-the-witcher/).

![Witcher Network](https://github.com/haciMMicah/graph2image/blob/main/resources/WitcherNetwork.png)

Lets say we would like the nodes of this network to be rearanged to fit within the contours of an image. 
The below image is a binary contour image of some different symbols from the Netflix Witcher adaptation.

![Netflix Witcher Symbols](https://github.com/haciMMicah/graph2image/blob/main/resources/witcherMedallion.jpg)

Running the graph2image script on the graphml file produced from this network with this command
```
python graph2image.py "../resources/WitcherNetwork.csv" "../resources/WitcherNetwork.graphml" "..resources/witcherMedallion.jpg" -H 1000 -w 1000 -R 100 -r 5 -a 2000 -o "./witcherPacked.graphml"
```
then gives us a new graphml file with the graph's nodes arranged to be packed into the contour image. The script takes a csv file 
representing an adjacency matrix of the graph (the csv output from a Gelphi file), a graphml representation of a graph (graphml output of a Gelphi file),
and an image file that represents the contour to be packed.

We can now import this graphml file into a graph visualization tool like Gelphi and get a visualization of this graph.
Below is without the edges of the graph displayed.

![Witcher Packed No Edges](https://github.com/haciMMicah/graph2image/blob/main/resources/WitcherPackedNoEdges.PNG)

And here it is with the edges displayed.

![Witcher Packed With Edges](https://github.com/haciMMicah/graph2image/blob/main/resources/WitcherPackedWithEdges.PNG)

This new graph doesn't tell us much in terms of node interaction. But can be used as a nice visualization tool to attract a reader's attention. 
Note: the node sizes of the new graph were increased in order to more adequately fill the negative space. This is a tunable parameter in the script with the -R and -r flags.


## How it works
graph2image uses openCV to threshold an image to a useful binary image of a contour. We then consider the black regions of the image as our contour to pack.
The packing algorithm is inspired by [Tyler Hobbs'](https://tylerxhobbs.com/essays/2016/a-randomized-approach-to-cicle-packing) randomized approach to circle packing 
a simple geometric shape. Instead of considering a circle intersection with simple shapes however, we consider any arbitrary 2-D contour and 
test the inclusion of 8 points of a circle's edge within the contour's boundaries without the need of algebraic collision detection. The algorithm then uses a greedy 
approach, trying to place the largest circle in the contour first. The algorithm also does not require placing all nodes, but keeps track of the nodes that are
placed and unplaced. Currently, the sizing of the nodes are determined by their outdegree, and the output sizing can be specified by a minimum and maximum 
circle radius.


