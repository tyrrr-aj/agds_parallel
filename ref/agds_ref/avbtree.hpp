#pragma once

#include <iterator>
#include <stdexcept>
#include <cmath>


// extern const float EPSILON;


template <typename T>
struct AVBInsertRes;


template<typename T>
struct AVBValue {
	T value;
	int count = 1;

	AVBValue* next = NULL;
	AVBValue* prev = NULL;

	AVBValue(const T& value) {
		this->value = value;
	}

	bool equals(T& other) {
		// return std::fabs(value - other) < EPSILON;
		return value == other.value;
	}

	void incrementCounter() {
		count++;
	}

	bool isLessThan(const T& other) {
		return value < other;
	}

	bool isLessThan(AVBValue<T>* other) {
		return value < other->value;
	}

	bool isGreaterThan(const T& other) {
		return value > other;
	}

	bool isGreaterThan(AVBValue<T>* other) {
		return value > other->value;
	}
};


template<typename T>
class AVBNode {
	AVBInsertRes<T> splitNode(AVBNode<T>* initiator, AVBNode<T>* passedRightNode, AVBValue<T>* newValue) {
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

		if (parent == NULL) {
			parent = new AVBNode<T>(this, newRightNode, newMiddleValue, parent);
			return AVBInsertRes<T>(newValue, true, parent);
		}
		else {
			return parent->insertUp(this, newRightNode, newMiddleValue);
		}
	}

	AVBInsertRes<T> insertUp(AVBNode<T>* initiator, AVBNode<T>* passedRightNode, AVBValue<T>* newValue) {
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

			return AVBInsertRes<T>(newValue, true);
		}
		else {
			return splitNode(initiator, passedRightNode, newValue);
		}
	}

	AVBInsertRes<T> splitLeaf(AVBValue<T>* newValue) {
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

		if (parent == NULL) {
			parent = new AVBNode<T>(this, newRightNeigh, middleValue, parent);
			return AVBInsertRes<T>(newValue, true, parent);
		}
		else {
			return parent->insertUp(this, newRightNeigh, middleValue);
		}
	}

	AVBInsertRes<T> insertNewValue(AVBValue<T>* newValue) {
		if (newValue->isLessThan(leftValue)) {	// newValue is leftmost element
			newValue->prev = leftValue->prev;
			newValue->next = leftValue;
		}
		else {
			if (rightValue != NULL && rightValue->isLessThan(newValue)) {	// newValue is rightmost element
				newValue->prev = rightValue;
				newValue->next = rightValue->next;
			}
			else {	// newValue is just next to the left element (is middle element or there is no right element)
				newValue->prev = leftValue;
				newValue->next = leftValue->next;
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

			return AVBInsertRes<T>(newValue, true);
		}
		else {
			return splitLeaf(newValue);
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

public:
	AVBNode<T>* parent = NULL;
	AVBNode<T>* leftChild = NULL;
	AVBNode<T>* middleChild = NULL;
	AVBNode<T>* rightChild = NULL;

	AVBValue<T>* leftValue;
	AVBValue<T>* rightValue = NULL;

	AVBInsertRes<T> insertDown(T value) {
		if (leftValue != NULL && leftValue->equals(value)) {
			leftValue->incrementCounter();
			return AVBInsertRes<T>(leftValue, false);
		}
		if (rightValue != NULL && rightValue->equals(value)) {
			rightValue->incrementCounter();
			return AVBInsertRes<T>(rightValue, false);
		}

		if (leftChild != NULL && leftValue->isGreaterThan(value)) {
			return leftChild->insertDown(value);
		}
		if (middleChild != NULL && rightValue->isGreaterThan(value)) {
			return middleChild->insertDown(value);
		}
		if (rightChild != NULL) {
			return rightChild->insertDown(value);
		}


		return insertNewValue(new AVBValue<T>(value));
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
struct AVBInsertRes {
	AVBNode<T>* newRoot;
	AVBValue<T>* insertedValue;
	bool valueIsNew;

	AVBInsertRes(AVBValue<T>* insertedValue, bool isValueNew) {
		newRoot = NULL;
		this->insertedValue = insertedValue;
		this->valueIsNew = isValueNew;
	}

	AVBInsertRes(AVBValue<T>* insertedValue, bool valueIsNew, AVBNode<T>* newRoot) {
		this->newRoot = newRoot;
		this->insertedValue = insertedValue;
		this->valueIsNew = valueIsNew;
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
	int size = 0;

	AVBTree() {}

	AVBValue<T>* insert(T value) {
		if (isEmpty()) {
			root = new AVBNode<T>(value);
			return root->leftValue;
		}
		else {
			AVBInsertRes<T> insertRes = root->insertDown(value);
			if (insertRes.newRoot != NULL) {
				root = insertRes.newRoot;
			}
			if (insertRes.valueIsNew) {
				size++;
			}
			return insertRes.insertedValue;
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

	bool isEmpty() {
		return root == NULL;
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

		AVBValue<T>* operator*() {
			return this->currentValue;
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
