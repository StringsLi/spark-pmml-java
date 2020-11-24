package main.java.com.string.pmml;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.PMML;
import org.jpmml.model.MetroJAXBUtil;
import org.jpmml.sparkml.PMMLBuilder;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.Serializable;

/**
 * 创建PMML模型文件
 */

public class PmmlCreator implements Serializable {
    private static final long serialVersionUID = 1344566L;

    public static void getPmmlFile(PipelineModel model, StructType schema,String outputPath){
        PMML pmml = new PMMLBuilder(schema, model).build();
        File file = new File(outputPath);
        try{
            OutputStream os = new FileOutputStream(file);
            MetroJAXBUtil.marshalPMML(pmml, os);
        }catch(Exception e){
            e.printStackTrace();
        }
    }
}
