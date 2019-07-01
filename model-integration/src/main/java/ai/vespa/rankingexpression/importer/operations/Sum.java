// Copyright 2018 Yahoo Holdings. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package ai.vespa.rankingexpression.importer.operations;

import ai.vespa.rankingexpression.importer.DimensionRenamer;
import ai.vespa.rankingexpression.importer.OrderedTensorType;
import com.yahoo.searchlib.rankingexpression.evaluation.DoubleValue;
import com.yahoo.searchlib.rankingexpression.evaluation.Value;
import com.yahoo.searchlib.rankingexpression.rule.ConstantNode;
import com.yahoo.searchlib.rankingexpression.rule.ExpressionNode;
import com.yahoo.searchlib.rankingexpression.rule.GeneratorLambdaFunctionNode;
import com.yahoo.tensor.Tensor;
import com.yahoo.tensor.TensorType;
import com.yahoo.tensor.functions.Generate;
import com.yahoo.tensor.functions.Reduce;
import com.yahoo.tensor.functions.ScalarFunctions;
import com.yahoo.tensor.functions.TensorFunction;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Optional;

public class Sum extends IntermediateOperation {

    private final AttributeMap attributeMap;
    private List<String> reduceDimensions;

    public Sum(String modelName, String nodeName, List<IntermediateOperation> inputs, AttributeMap attributeMap) {
        super(modelName, nodeName, inputs);
        this.attributeMap = attributeMap;
    }

    @Override
    protected OrderedTensorType lazyGetType() {
        if ( ! allInputTypesPresent(2)) return null;

        IntermediateOperation reductionIndices = inputs.get(1);
        if ( ! reductionIndices.getConstantValue().isPresent()) {
            throw new IllegalArgumentException("Sum in " + name + ": Reduction indices must be a constant.");
        }
        Tensor indices = reductionIndices.getConstantValue().get().asTensor();
        reduceDimensions = new ArrayList<>();

        OrderedTensorType inputType = inputs.get(0).type().get();
        for (Iterator<Tensor.Cell> cellIterator = indices.cellIterator(); cellIterator.hasNext();) {
            Tensor.Cell cell = cellIterator.next();
            int dimensionIndex = cell.getValue().intValue();
            if (dimensionIndex < 0) {
                dimensionIndex = inputType.dimensions().size() - dimensionIndex;
            }
            reduceDimensions.add(inputType.dimensions().get(dimensionIndex).name());
        }
        return reducedType(inputType, shouldKeepDimensions());
    }

    // optimization: if keepDims and one reduce dimension that has size 1: same as identity.

    @Override
    protected TensorFunction lazyGetFunction() {
        if ( ! allInputTypesPresent(2)) return null;

        TensorFunction inputFunction = inputs.get(0).function().get();
        TensorFunction output = new Reduce(inputFunction, Reduce.Aggregator.sum, reduceDimensions);
        if (shouldKeepDimensions()) {
            // multiply with a generated tensor created from the reduced dimensions
            TensorType.Builder typeBuilder = new TensorType.Builder(resultValueType());
            for (String name : reduceDimensions) {
                typeBuilder.indexed(name, 1);
            }
            TensorType generatedType = typeBuilder.build();
            ExpressionNode generatedExpression = new ConstantNode(new DoubleValue(1));
            Generate generatedFunction = new Generate(generatedType,
                    new GeneratorLambdaFunctionNode(generatedType, generatedExpression).asLongListToDoubleOperator());
            output = new com.yahoo.tensor.functions.Join(output, generatedFunction, ScalarFunctions.multiply());
        }
        return output;
    }

    @Override
    public void renameDimensions(DimensionRenamer renamer) {
        super.renameDimensions(renamer);
        List<String> renamedDimensions = new ArrayList<>(reduceDimensions.size());
        for (String name : reduceDimensions) {
            Optional<String> newName = renamer.dimensionNameOf(name);
            if (!newName.isPresent()) {
                return;  // presumably, already renamed
            }
            renamedDimensions.add(newName.get());
        }
        reduceDimensions = renamedDimensions;
    }

    private boolean shouldKeepDimensions() {
        Optional<Value> keepDims = attributeMap.get("keep_dims");
        return keepDims.isPresent() && keepDims.get().asBoolean();
    }

    private OrderedTensorType reducedType(OrderedTensorType inputType, boolean keepDimensions) {
        OrderedTensorType.Builder builder = new OrderedTensorType.Builder(resultValueType());
        for (TensorType.Dimension dimension: inputType.type().dimensions()) {
            if ( ! reduceDimensions.contains(dimension.name())) {
                builder.add(dimension);
            } else if (keepDimensions) {
                builder.add(TensorType.Dimension.indexed(dimension.name(), 1L));
            }
        }
        System.out.println("----------> Sum input type is " + inputType + ", keepDimensions: " + keepDimensions + ", result: " + builder.build());
        return builder.build();
    }

    @Override
    public String toString() {
        return "Sum(" + asString(inputs().get(0).type()) + ", " + asString(inputs().get(1).type()) + ", " + reduceDimensions + ")";
    }

    @Override
    public String toFullString() {
        return "\t" + lazyGetType() + ":\tSum[keep_dims=" + shouldKeepDimensions() + "](" +
               inputs().get(0).toFullString() + ", " + inputs().get(1).toFullString() + ", " + reduceDimensions + ")";
    }

}
