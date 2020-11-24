package main.java.com.string.pmml.serving;

import org.dmg.pmml.FieldName;
import java.util.HashMap;
import java.util.Map;

public class JPMMLModelServing {
    public static void main(String[] args){

        JavaModelServer javaModelServer = new JavaModelServer("D:\\data\\pmml/DT1.pmml");
        HashMap<String, Object> featureMap = new HashMap<String, Object>();
//        "age","hypertension","heart_disease","bmi_full"
        featureMap.put("age", 18);
        featureMap.put("hypertension", 52);
        featureMap.put("heart_disease", 16);
        featureMap.put("bmi_full", 25);

        Map<FieldName, ?> result = javaModelServer.forecast(featureMap);

//        for (Map.Entry<FieldName, ?> field : result.entrySet()){
//            System.out.println(field.getKey().getValue() + ":\t" +  field.getValue());
//        }

        for(int i = 0 ; i < result.size(); i++){
            System.out.println(result);
        }
    }
}
