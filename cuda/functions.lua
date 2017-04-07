function getMean(inputs,targets,lenet)
    diff = torch.abs(lenet:forward(inputs) - targets)
    mean = diff:mean()
    return mean
end

function performanceEvaluator(trainset,lenet,eList)
  items = eList or {8,14,32,49,53,58,65,68,69,70} 
  length = #eList or (#eList)[1] or 10
  means = torch.Tensor(length)
  for i=1,length do
    inputs = trainset.images[items[i]]
    targets = torch.Tensor(1,32,32):fill(trainset.labels[items[i]]*0.3)
    means[i] = getMean(inputs,targets,lenet)*100/(trainset.labels[items[i]]*0.3)
    print('Image '.. items[i]..' | Label '.. string.format('%2d',trainset.labels[items[i]])  .. ' | Sigma '..string.format('%2.1f', trainset.labels[items[i]]*0.3) ..' | MSE ' .. string.format('%1.8f',criterion:forward(lenet:forward(inputs),targets)) .. ' | Percent Mean ' .. means[i])
  end
  print('Means of means: ' .. means:mean())
end


function classPerformance(trainset,lenet,eList,single)
  items = eList or {8,14,32,49,53,58,65,68,69,70} 
  length = (#eList)[1] or 10
  acc = torch.Tensor(10):fill(0)
  hist = torch.Tensor(10):fill(0)
  for i=1,length do
    inputs = trainset.images[items[i]]:cuda()
    if single then
      targets = torch.Tensor(1,1,1):fill(trainset.labels[items[i]]*0.3):cuda()
    else
      targets = torch.Tensor(1,32,32):fill(trainset.labels[items[i]]*0.3):cuda()
    end
    acc[trainset.labels[items[i]]] = acc[trainset.labels[items[i]]]+ getMean(inputs,targets,lenet)*100/(trainset.labels[items[i]]*0.3)
    hist[trainset.labels[items[i]]] = hist[trainset.labels[items[i]]] + 1
  end
  print(torch.cdiv(acc,hist))
  print(torch.cdiv(acc,hist):mean())
end

function trainSingle(trainset,lenet,lr,item)
  params,grad_params = lenet:getParameters();
  inputs = trainset.images[item]:cuda()
  targets = torch.Tensor(1,1,1):fill(trainset.labels[item]*0.3):cuda()
  grad_params:zero();
  outputs = lenet:forward(inputs);
  currentError = currentError + criterion:forward(outputs, targets);
  df_do = criterion:backward(outputs, targets);
  lenet:backward(inputs, df_do);
  -- params:add(-lr,grad_params);
  -- sgd(params,grad_params,lr)
  -- rmsprop(params,grad_params,lr,0.99,1e-8,config)
  -- adam(params,grad_params,lr,0.9,0.999,1e-7,config)
  adagrad(params,grad_params,lr,1e-7,config)
end