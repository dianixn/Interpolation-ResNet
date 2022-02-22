function Pruned_NN = Pruning(NN, Pruning_Location)

Transfer_NN = NN.saveobj;

for i = 1 : size(Pruning_Location, 1)
    Transfer_NN.Layers(Pruning_Location{i}(1)).Weights(Pruning_Location{i}(2), Pruning_Location{i}(3), Pruning_Location{i}(4), Pruning_Location{i}(5)) = 0;
end

% Transfer to DAG Net
Pruned_NN = NN.loadobj(Transfer_NN);