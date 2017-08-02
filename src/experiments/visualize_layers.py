import settings_stats as ss
import test_network as tn
import Weightstore as ws

if __name__ == "__main__":


   settings_file_path = "settings_to_test.txt"
   settings = list(map(lambda x : str(x.guid), ss.load_settings_from_file(settings_file_path)))

   layers = [0]

   for s in settings:
      net = ws.get_net(s)

      for l in layers:
         tn.visualize_weights(net,l)
