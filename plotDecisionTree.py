from decisionTree import createDataSet
from decisionTree import createTree
import matplotlib.pyplot as plt

dataSet, labels = createDataSet()

#决策树
decisionTree = createTree(dataSet, labels)

#决策节点样式为锯齿框
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
#叶结点样式
leafNode = dict(boxstyle="round4", fc="0.8")
#箭头样式
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords="axes fraction",xytext=centerPt,\
    textcoords="axes fraction", va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)
    
def createPlot():
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode("decisionNode", (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('leafNode', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

def getNumLeafs(myTree):
    numLeafs = 0
    for key in myTree:
        if isinstance(myTree[key], dict):
            numLeafs += getNumLeafs(myTree[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    if not isinstance(myTree, dict):
        #因为是字典，所以退一层
        return -1
    else:
        valueList = list(map(lambda x:myTree[x], list(myTree)))
        maxValue = max(list(map(getTreeDepth, valueList)))
        return maxValue + 1

#两点之间的文本
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] + cntrPt[0])/2
    yMid = (parentPt[1] + cntrPt[1])/2
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree)[0]
    cntrPt = (plotTree.xOff + (1+numLeafs)/2/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1/plotTree.totalD
    for key in secondDict:
        if isinstance(secondDict[key], dict):
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1/plotTree.totalD
    
def createPlot(inTree):
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = getNumLeafs(inTree)
    plotTree.totalD = getTreeDepth(inTree)
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1
    plotTree(inTree, (0.5,1), "")
    plt.show()

createPlot(decisionTree)