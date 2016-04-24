-----------------------------------------------------------------------------------------
-- This program utilizes data from the 1914 - 2014 Major League Baseball seasons 
-- in order to develop an algorithm that predicts the number of wins for a given
-- team in the 2015 season based on several different indicators of success.

-- There are 16 different features that will be used as the inputs to the neural 
-- network and the output will be a value that represents the number of wins. 

-- Input features: Runs, At Bats, Hits, Doubles, Triples, Homeruns, Walks, Strikeouts,
-- Stolen Bases, Runs Allowed, Earned Runs, Earned Run Average (ERA), Shutouts, 
-- Saves, and Errors

-- Output: Number of predicted wins 

-- General Approach: For predicting the number of wins a baseball team will attain 
-- based on the given input features, a linear regression approach is neccessary. 
-- The data is converted from a CSV file into a tensor. This is done through
-- csv2tensor. More info can be found here (https://github.com/willkurt/csv2tensor).

-- Future Work: 

-- Generating more features that can contrbute to a team's success over a season.
-- New neural network architectures that will minimize the error between predicted and actual values. 
-- Adjusting hyperparameters such as the learning rate.

-- Dataset: http://www.seanlahman.com/baseball-archive/statistics/
-----------------------------------------------------------------------------------------

-- Optim is a package that needs to be installed seperately
-- torch-pkg install optim

require 'torch'
require 'nn'
require 'optim'

-----------------------------------------------------------------------------------------
-- 1. Data Formatting

-- The data used by this program is placed into 2 csv files. One file contains the 
-- training data (stats from 1914 - 2014 MLB seasons) while one contains the test
-- set from the 2015 MLB season. The data from both of these CSV files is extracted
-- into two tensors. The training data is a 2184 x 17 dimensional tensor. 2184 represents
-- the number of seasons/teams we have for training. The number of wins will be in the 
-- 17th column, while the input features are placed in cols 1-16. The test data is a 
-- 30 x 17 dimensional tensor, as this will contain every MLB team's 2015 season.

csv2tensor = require 'csv2tensor'
train_data = csv2tensor.load("MLB_1914_2014.csv")
test_data = csv2tensor.load("MLB_2015.csv")

-----------------------------------------------------------------------------------------
-- 2. Neural Network Architecture

model = nn.Sequential()
model:add(nn.Linear(16, 1))
criterion = nn.MSECriterion()

-----------------------------------------------------------------------------------------
-- 3. Evaluation Function

x, dl_dx = model:getParameters()

feval = function()
	_nidx_ = (_nidx_ or 0) + 1
	if _nidx_ > (#train_data)[1] then _nidx_ = 1 end

	local training_example = train_data[_nidx_]
	local target_value = training_example[{ {17} }]
	local inputs = training_example[{ {1,16} }]
	dl_dx:zero()
	local loss = criterion:forward(model:forward(inputs), target_value)
	model:backward(inputs, criterion:backward(model.output, target_value))
	return loss, dl_dx
end

-----------------------------------------------------------------------------------------
-- 4. Setting Hyperparameters

sgd_params = {
	learningRate = 1e-8,
}

-----------------------------------------------------------------------------------------
-- 5. Training 

for i = 1,1e5 do
	current_loss = 0
	for i = 1, (#train_data)[1] do
		_,fs = optim.sgd(feval, x, sgd_params)
		current_loss = current_loss + fs[1]
	end
	current_loss = current_loss / (#data)[1]
	print(current_loss)
end

-----------------------------------------------------------------------------------------
-- 6. Testing

print ('PREDICTIONS FOR 2015')

for i = 1, (#test_data)[1] do
	local pred = (model:forward(test_data[i][{{1,16}}]))
	print (pred[1])
end