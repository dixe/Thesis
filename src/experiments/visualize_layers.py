import settings_stats as ss
import test_network as tn
import Weightstore as ws

if __name__ == "__main__":


   settings_file_path = "settings_to_test.txt"
   settings = list(map(lambda x : str(x.guid), ss.load_settings_from_file(settings_file_path)))

   layers = [0]

   for s in settings:
      setting = ws.get_settings(s)


      if 'simple_model' == setting.model_name:
         import simple_model as sm
      elif 'simple_model_7_5_5' == setting.model_name:
         import simple_model_7_5_5 as sm
      elif 'simple_model_7_fully_drop' == setting.model_name:
         import simple_model_7_fully_drop as sm
      elif 'simple_model_7_2_layer' == setting.model_name:
         import simple_model_7_2_layer as sm
      elif 'simple_model_7_nomax' == setting.model_name:
         import simple_model_7_nomax as sm
      elif 'simple_model_min_7_drop' == setting.model_name:
         import simple_model_min_7_drop as sm
      elif 'simple_model_min_7' == setting.model_name:
         import simple_model_min_7 as sm

      net = sm.get_net(setting)
      for l in layers:
         tn.visualize_weights(net,l)
