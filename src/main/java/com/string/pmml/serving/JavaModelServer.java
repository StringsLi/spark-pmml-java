package main.java.com.string.pmml.serving;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.Evaluator;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.InputField;
import org.jpmml.evaluator.ModelEvaluatorFactory;
import org.xml.sax.SAXException;

import javax.xml.bind.JAXBException;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * java 读取PMML模型文件
 */

public class JavaModelServer {

    String modelPath;

    Evaluator model;

    public JavaModelServer(String modelPath){
        this.modelPath = modelPath;
    }

    public void loadModel(){
        PMML pmml = new PMML();
        InputStream inputStream = null;
        try {
            inputStream = new FileInputStream(modelPath);
        } catch (IOException e) {
            e.printStackTrace();
        }

        InputStream is = inputStream;
        try {
            pmml = org.jpmml.model.PMMLUtil.unmarshal(is);
        } catch (SAXException e1) {
            e1.printStackTrace();
        } catch (JAXBException e1) {
            e1.printStackTrace();
        }

        ModelEvaluatorFactory modelEvaluatorFactory = ModelEvaluatorFactory.newInstance();
        this.model = modelEvaluatorFactory.newModelEvaluator(pmml);
        this.model.verify();
    }

    public Map<FieldName, ?> forecast(Map<String, ?> featureMap){
        if (this.model == null){
            loadModel();
        }
        if (featureMap == null){
            System.err.println("features is null");
            return null;
        }

        List<InputField> inputFields = this.model.getInputFields();
        Map<FieldName, FieldValue> pmmlFeatureMap = new LinkedHashMap<FieldName, FieldValue>();
        for (InputField inputField : inputFields){
            if (featureMap.containsKey(inputField.getName().getValue())) {
                Object value = featureMap.get(inputField.getName().getValue());
                pmmlFeatureMap.put(inputField.getName(), inputField.prepare(value));
            }else{
                System.err.println("lack of feature: " + inputField.getName().getValue());
                return null;
            }
        }
        return this.model.evaluate(pmmlFeatureMap);
    }
}
