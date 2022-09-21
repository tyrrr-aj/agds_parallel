#pragma once

#include <iterator>
#include <stdexcept>


extern float EPSILON;

template<typename T>
struct AVBInsertResult;


template<typename T>
struct AVBValue {
	T value;
	int count = 1;
	int offset = 0;

	AVBValue* next = NULL;
	AVBValue* prev = NULL;

	AVBValue(T value) {
		this->value = value;
	}

	bool equals(T other) {
		return std::fabs(value - other) < EPSILON;
	}

	void incrementCounter() {
		count++;
	}

	bool isLessThan(T other) {
		return value < other;
	}

	bool isLessThan(AVBValue<T>* other) {
		return value < other->value;
	}

	bool isGreaterThan(T other) {
		return value > other;
	}

	bool isGreaterThan(AVBValue<T>* other) {
		return value > other->value;
	}
};


template<typename T>
class AVBNode {
	bool nValuesInBranchAlreadySet = false;

	AVBInsertResult<T> splitNode(AVBNode<T>* initiator, AVBNode<T>* passedRightNode, AVBValue<T>* newValue) {
		AVBValue<T>* newLeftValue, * newMiddleValue, * newRightValue;
		AVBNode<T>* newLeftmostChild, * newLeftMiddleChild, * newRightMiddleChild, * newRightmostChild;

		if (initiator == leftChild)
		{
			newLeftValue = newValue;
			newMiddleValue = leftValue;
			newRightValue = rightValue;
			newLeftmostChild = initiator;
			newLeftMiddleChild = passedRightNode;
			newRightMiddleChild = middleChild;
			newRightmostChild = rightChild;
		}
		else if (initiator == middleChild) {
			newLeftValue = leftValue;
			newMiddleValue = newValue;
			newRightValue = rightValue;
			newLeftmostChild = leftChild;
			newLeftMiddleChild = initiator;
			newRightMiddleChild = passedRightNode;
			newRightmostChild = rightChild;
		}
		else {	// initiator == rightChild)
			newLeftValue = leftValue;
			newMiddleValue = rightValue;
			newRightValue = newValue;
			newLeftmostChild = leftChild;
			newLeftMiddleChild = middleChild;
			newRightMiddleChild = initiator;
			newRightmostChild = passedRightNode;
		}

		AVBNode<T>* newRightNode = new AVBNode<T>(newRightMiddleChild, newRightmostChild, newRightValue, parent);

		leftValue = newLeftValue;
		rightValue = NULL;

		setLeftChild(newLeftmostChild);
		setMiddleChild(NULL);
		setRightChild(newLeftMiddleChild);

		nValuesInBranch = 1 + (leftChild != NULL ? leftChild->nValuesInBranch : 0) + (rightChild != NULL ? rightChild->nValuesInBranch : 0);
		nValuesInBranchAlreadySet = true;

		if (parent == NULL) {
			parent = new AVBNode<T>(this, newRightNode, newMiddleValue, parent);
			return AVBInsertResult<T>(parent, newValue, false);
		}
		else {
			return parent->insertUp(this, newRightNode, newMiddleValue);
		}
	}

	AVBInsertResult<T> insertUp(AVBNode<T>* initiator, AVBNode<T>* passedRightNode, AVBValue<T>* newValue) {
		AVBValue<T>* newLeftValue, * newRightValue;
		AVBNode<T>* newLeftChild, * newMiddleChild, * newRightChild;

		if (rightValue == NULL) {
			if (initiator == leftChild) {
				newLeftValue = newValue;
				newRightValue = leftValue;
				newLeftChild = initiator;
				newMiddleChild = passedRightNode;
				newRightChild = rightChild;
			}
			else {
				newLeftValue = leftValue;
				newRightValue = newValue;
				newLeftChild = leftChild;
				newMiddleChild = initiator;
				newRightChild = passedRightNode;
			}

			leftValue = newLeftValue;
			rightValue = newRightValue;
			setLeftChild(newLeftChild);
			setMiddleChild(newMiddleChild);
			setRightChild(newRightChild);

			nValuesInBranch++;
			nValuesInBranchAlreadySet = true;

			return AVBInsertResult<T>(NULL, newValue, false);
		}
		else {
			return splitNode(initiator, passedRightNode, newValue);
		}
	}

	AVBInsertResult<T> splitLeaf(AVBValue<T>* newValue) {
		AVBValue<T>* leftmostValue, * middleValue, * rightmostValue;

		if (newValue->isLessThan(leftValue)) {
			leftmostValue = newValue;
			middleValue = leftValue;
			rightmostValue = rightValue;
		}
		else if (newValue->isLessThan(rightValue)) {
			leftmostValue = leftValue;
			middleValue = newValue;
			rightmostValue = rightValue;
		}
		else {
			leftmostValue = leftValue;
			middleValue = rightValue;
			rightmostValue = newValue;
		}

		leftValue = leftmostValue;
		rightValue = NULL;

		AVBNode<T>* newRightNeigh = new AVBNode<T>(rightmostValue, parent);

		nValuesInBranch = 1;

		if (parent == NULL) {
			parent = new AVBNode<T>(this, newRightNeigh, middleValue, parent);
			parent->nValuesInBranch = 3;
			return AVBInsertResult<T>(parent, newValue, false);
		}
		else {
			return parent->insertUp(this, newRightNeigh, middleValue);
		}
	}

	AVBInsertResult<T> insertNewValue(AVBValue<T>* newValue, int nPreviousValues) {
		int newValueIndex;

		if (newValue->isLessThan(leftValue)) {	// newValue is leftmost element
			newValue->prev = leftValue->prev;
			newValue->next = leftValue;

			newValueIndex = getLeftValueIndex(nPreviousValues);
		}
		else {
			if (rightValue != NULL && rightValue->isLessThan(newValue)) {	// newValue is rightmost element
				newValue->prev = rightValue;
				newValue->next = rightValue->next;

				newValueIndex = getRightValueIndex(nPreviousValues) + 1;
			}
			else {	// newValue is just next to the left element (is middle element or there is no right element)
				newValue->prev = leftValue;
				newValue->next = leftValue->next;

				newValueIndex = getRightValueIndex(nPreviousValues);
			}
		}

		if (newValue->prev != NULL) newValue->prev->next = newValue;
		if (newValue->next != NULL) newValue->next->prev = newValue;

		if (rightValue == NULL) {	// newValue can be inserted to the current leaf (there was only one value stored)
			if (newValue->isLessThan(leftValue)) {
				rightValue = leftValue;
				leftValue = newValue;
			}
			else {
				rightValue = newValue;
			}

			nValuesInBranch++;

			return AVBInsertResult<T>(NULL, newValue, false, newValueIndex);
		}
		else {
			AVBInsertResult<T> res = splitLeaf(newValue);
			res.index = newValueIndex;
			return res;
		}
	}

	void setLeftChild(AVBNode<T>* newChild) {
		leftChild = newChild;
		if (newChild != NULL) {
			newChild->parent = this;
		}
	}

	void setMiddleChild(AVBNode<T>* newChild) {
		middleChild = newChild;
		if (newChild != NULL) {
			newChild->parent = this;
		}
	}

	void setRightChild(AVBNode<T>* newChild) {
		rightChild = newChild;
		if (newChild != NULL) {
			newChild->parent = this;
		}
	}

	bool isLeaf() {
		return leftChild == NULL && middleChild == NULL && rightChild == NULL;
	}

	int getLeftValueIndex(int nPreviousValues) {
		return nPreviousValues + (leftChild != NULL ? leftChild->nValuesInBranch : 0);
	}

	int getRightValueIndex(int nPreviousValues) {
		return getLeftValueIndex(nPreviousValues) + (middleChild != NULL ? middleChild->nValuesInBranch : 0) + 1;
	}

public:
	int nValuesInBranch = 1;

	AVBNode<T>* parent = NULL;
	AVBNode<T>* leftChild = NULL;
	AVBNode<T>* middleChild = NULL;
	AVBNode<T>* rightChild = NULL;

	AVBValue<T>* leftValue;
	AVBValue<T>* rightValue = NULL;

	AVBInsertResult<T> insertDown(T value, int nPreviousValues) {
		if (leftValue != NULL && leftValue->equals(value)) {
			leftValue->incrementCounter();
			return AVBInsertResult<T>(NULL, leftValue, true, getLeftValueIndex(nPreviousValues));
		}
		if (rightValue != NULL && rightValue->equals(value)) {
			rightValue->incrementCounter();
			return AVBInsertResult<T>(NULL, rightValue, true, getRightValueIndex(nPreviousValues));
		}

		if (isLeaf()) {
			return insertNewValue(new AVBValue<T>(value), nPreviousValues);
		}
		else {
			AVBInsertResult<T> insertRes;
			nValuesInBranchAlreadySet = false;

			if (leftChild != NULL && leftValue->isGreaterThan(value)) {
				insertRes = leftChild->insertDown(value, nPreviousValues);
			}
			else if (middleChild != NULL && rightValue->isGreaterThan(value)) {
				insertRes = middleChild->insertDown(value, nPreviousValues + leftChild->nValuesInBranch + 1);
			}
			else if (rightChild != NULL) {
				insertRes = rightChild->insertDown(value, nPreviousValues + leftChild->nValuesInBranch + (middleChild != NULL ? middleChild->nValuesInBranch + 1 : 0) + 1);
			}

			if (!insertRes.valueAlreadyInTree && !nValuesInBranchAlreadySet) {
				nValuesInBranch++;
			}

			return insertRes;
		}
	}

	AVBValue<T>* getMinTreeValue() {
		return isLeaf() ? leftValue : leftChild->getMinTreeValue();
	}

	AVBValue<T>* getMaxTreeValue() {
		if (isLeaf()) {
			return rightValue != NULL ? rightValue : leftValue;
		}
		else {
			return rightChild->getMaxTreeValue();
		}
	}

	int getIndex(T searchedValue, int nPreviousValues) {
		if (leftValue->equals(searchedValue)) {
			return getLeftValueIndex(nPreviousValues);
		}
		if (rightValue != NULL && rightValue->equals(searchedValue)) {
			return getRightValueIndex(nPreviousValues);
		}

		if (leftChild != NULL && leftValue->isGreaterThan(searchedValue)) {
			return leftChild->getIndex(searchedValue, nPreviousValues);
		}
		if (rightValue == NULL && rightChild != NULL && leftValue->isLessThan(searchedValue)) {
			return rightChild->getIndex(searchedValue, nPreviousValues + leftChild->nValuesInBranch + 1);
		}
		if (middleChild != NULL && rightValue->isGreaterThan(searchedValue)) {
			return middleChild->getIndex(searchedValue, nPreviousValues + leftChild->nValuesInBranch + 1);
		}
		if (rightChild != NULL && rightValue != NULL && rightValue->isLessThan(searchedValue)) {
			return rightChild->getIndex(searchedValue, nPreviousValues + leftChild->nValuesInBranch + middleChild->nValuesInBranch + 2);
		}

		throw std::invalid_argument("Value searched with getIndex() method not found in the tree");
	}

	AVBNode(T value) {
		// regular constructor
		leftValue = new AVBValue<T>(value);
	}

	AVBNode(AVBValue<T>* value, AVBNode<T>* parent) {
		leftValue = value;
		this->parent = parent;
	}

	AVBNode(AVBNode<T>* leftChild, AVBNode<T>* rightChild, AVBValue<T>* ownValue, AVBNode<T>* parent) {
		// new root/middle node constructor
		leftValue = ownValue;
		this->parent = parent;
		setLeftChild(leftChild);
		setRightChild(rightChild);
		nValuesInBranch = 1 + (leftChild != NULL ? leftChild->nValuesInBranch : 0) + (rightChild != NULL ? rightChild->nValuesInBranch : 0);
	}

	~AVBNode() {
		if (leftChild != NULL) delete leftChild;
		if (middleChild != NULL) delete middleChild;
		if (rightChild != NULL) delete rightChild;

		if (leftValue != NULL) delete leftValue;
		if (rightValue != NULL) delete rightValue;
	}
};


template<typename T>
struct AVBInsertResult {
	AVBNode<T>* newRoot;
	AVBValue<T>* insertedValue;
	bool valueAlreadyInTree;
	int index;

	AVBInsertResult() {
		newRoot = NULL;
		insertedValue = NULL;
		valueAlreadyInTree = false;
		index = -1;
	}

	AVBInsertResult(AVBNode<T>* newRoot, AVBValue<T>* insertedValue, bool valueAlreadyInTree) {
		this->newRoot = newRoot;
		this->insertedValue = insertedValue;
		this->valueAlreadyInTree = valueAlreadyInTree;
		index = -1;
	}

	AVBInsertResult(AVBNode<T>* newRoot, AVBValue<T>* insertedValue, bool valueAlreadyInTree, int index) {
		this->newRoot = newRoot;
		this->insertedValue = insertedValue;
		this->valueAlreadyInTree = valueAlreadyInTree;
		this->index = index;
	}
};


template<typename T>
struct TreeInsertResult {
	AVBValue<T>* insertedValue = NULL;
	int index = -1;
	bool wasNewVnCreated = false;

	TreeInsertResult() {};
	TreeInsertResult(AVBValue<T>* insertedValue, int index, bool wasNewVnCreated) {
		this->insertedValue = insertedValue;
		this->index = index;
		this->wasNewVnCreated = wasNewVnCreated;
	}
	TreeInsertResult(AVBInsertResult<T>& internalRes) {
		insertedValue = internalRes.insertedValue;
		index = internalRes.index;
		wasNewVnCreated = !internalRes.valueAlreadyInTree;
	}
};


template<typename T>
class AVBTree {
	AVBNode<T>* root = NULL;

	AVBValue<T>* minValue() {
		return root != NULL ? root->getMinTreeValue() : NULL;
	}

	AVBValue<T>* maxValue() {
		return root != NULL ? root->getMaxTreeValue() : NULL;
	}


public:
	AVBTree() {}

	TreeInsertResult<T> insert(T value) {
		if (isEmpty()) {
			root = new AVBNode<T>(value);
			return TreeInsertResult<T>(root->leftValue, 0, true);
		}
		else {
			AVBInsertResult<T> result = root->insertDown(value, 0);
			if (result.newRoot != NULL) {
				root = result.newRoot;
			}
			result.insertedValue->offset = size() - 1;
			return TreeInsertResult<T>(result);
		}
	}

	T min() {
		AVBValue<T>* result = minValue();
		if (result == NULL) {
			throw std::invalid_argument("min() method called on empty AVBTree");
		}

		return result->value;
	}

	T max() {
		AVBValue<T>* result = maxValue();
		if (result == NULL) {
			throw std::invalid_argument("max() method called on empty AVBTree");
		}

		return result->value;
	}

	int getIndex(T value) {
		if (root == NULL) {
			throw std::invalid_argument("getIndex() method called on empty AVBTree");
		}

		return root->getIndex(value, 0);
	}

	bool isEmpty() {
		return root == NULL;
	}

	int size() {
		return isEmpty() ? 0 : root->nValuesInBranch;
	}


	struct Iterator {
		using iterator_category = std::bidirectional_iterator_tag;

		Iterator() { currentValue = NULL; }
		Iterator(AVBValue<T>* avbValue) { currentValue = avbValue; }

		Iterator& operator++() {
			if (currentValue != NULL) {
				currentValue = currentValue->next;
			}

			return *this;
		}

		Iterator operator++(int) {
			Iterator tempIter = *this;
			++* this;
			return tempIter;
		}

		Iterator& operator--() {
			if (currentValue != NULL) {
				currentValue = currentValue->prev;
			}

			return *this;
		}

		Iterator operator--(int) {
			Iterator tempIter = *this;
			--* this;
			return tempIter;
		}

		bool operator==(const Iterator& other) {
			return this->currentValue == other.currentValue;
		}

		bool operator!=(const Iterator& other) {
			return this->currentValue != other.currentValue;
		}

		T operator*() {
			return this->currentValue->value;
		}

	private:
		AVBValue<T>* currentValue;
	};

	Iterator begin() {
		return Iterator(minValue());
	}

	Iterator end() {
		return Iterator();
	}

	~AVBTree() {
		if (root != NULL) {
			delete root;
		}
	}
};
