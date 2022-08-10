// Preston Walraven 
// CACI BITS-IRAD Beholder
// https://gist.github.com/danielbierwirth/0636650b005834204cb19ef5ae6ccedb

using System;
using System.IO;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using UnityEngine;
using System.Net;
using System.Net.Sockets;

public class Dispatcher : MonoBehaviour
{
    public Queue<Action> _lock = new Queue<Action>();

    /// <summary>
    /// On update loop executes lambdas held by the dispatcher.
    /// </summary>
    void Update()
    {
        lock (_lock)
        {
            while (_lock.Count != 0) _lock.Dequeue().Invoke();
        }
    }

    /// <summary>
    /// Schedules the specified action to be performed by the dispatcher.
    /// </summary>
    /// <param name="a">The lambda to add the the event queue.</param>
    public void Invoke(Action a)
    {
        lock (_lock)
        {
            _lock.Enqueue(a);
        }
    }
}

/// <summary>
/// Establishes a TCP connection on a background thread, passes the parsed 
/// JSON to be added to the scene in the main thread.
/// </summary>
public class TCP : MonoBehaviour
{
    public GameObject car;
    public GameObject person;
    public Vector3 holo_ecef;
    public Vector3 cam_ecef;

    private Dispatcher dispatcher; // for updating the scene via the main thread
    private JsonData json_obj; 
    private Dictionary<string, GameObject> targetDictionary;
    private Vector3 labelOffset = new Vector3(0f, 1f, 0f);
    private Thread clientReceiveThread;

    /// <summary>
    /// Represents each unique object in the scene to parse. 
    /// </summary>
    [Serializable]
    public class JsonObject
    {
        public int track_id;
        public string label_name;
        //public Tuple<float, float, float> point; // lat, lon, altitude
        public List<float> point;
    }

    /// <summary>
    /// Outer list of targets and other metadata.
    /// </summary>
    [Serializable]
    public class JsonData
    {
        public List<JsonObject> tracks;
    }

    // Use this for initialization 	
    void Start()
    {
        // setup unity
        car.SetActive(false);
        person.SetActive(false);
        json_obj = new JsonData();
        // need to fill with data to start correctly
        JsonUtility.FromJsonOverwrite("{\"frame_num\": 210, \"tracks\": [{\"roi\": [537, 431, 590, 461], \"track_id\": 999, \"tracklet_id\": 999, \"label\": 1, \"label_name\": \"connected\", \"confidence\": 0.9053571564810616, \"top_attributes\": null, \"metadata\": null, \"point\": [0.0, 0.0, 80.0]}]}", json_obj);
        dispatcher = gameObject.AddComponent<Dispatcher>();
        targetDictionary = new Dictionary<string, GameObject>();

        // setup hololens
        holo_ecef = APIGetHoloLensLocation();
        ConnectToTcpServer();
    }

    /// <summary>
    /// An example API call for getting the ECEF coordinates of this device (Hololens).
    /// </summary>
    public Vector3 APIGetHoloLensLocation()
    {
        // in the future, this might need to be converted from whatever incoming coordinate standard
        // to ecef

        // Lat, Lon, Alt of (0,0,0) => this vector in ECEF
        // Unity needs y and z switched
        //return new Vector3((float)6378137.0, (float)0.0, (float)0.0);
        return new Vector3((float)0.0, (float)0.0, (float)0.0);
    }

    /// <summary> 	
    /// Setup socket connection. Thread to loop and listen.	
    /// </summary> 	
    private void ConnectToTcpServer()
    {
        try
        {
            clientReceiveThread = new Thread(new ThreadStart(ListenForData));
            clientReceiveThread.IsBackground = true;
            clientReceiveThread.Start();
        }
        catch (Exception e)
        {
            System.Diagnostics.Debug.WriteLine("Connection Exception: " + e);
        }
    }

    /// <summary>
    /// Adds the parsed targets to the Unity Scene, updates their position, or deletes them.
    /// </summary>
    /// <param name="tracks">The list of parsed targets to add to the scene.</param>
    public void UpdateScene(List<JsonObject> tracks)
    {
        lock (targetDictionary)
        {
            print("Updating scene.");
            foreach (JsonObject target in tracks)
            {
                //print($"{obj.cls_id} [{obj.posn[0]}, {obj.posn[1]}, {obj.posn[2]}]");

                string cls_id = $"{target.label_name}_{target.track_id}";

                Vector3 newPosn = new Vector3(
                    target.point[0] - holo_ecef[0],
                    target.point[1] - holo_ecef[1], // TODO maybe switch 2 and 3 due to unity reference frame?
                    target.point[2] - holo_ecef[2]);

                GameObject namePlate;

                if (targetDictionary.ContainsKey(cls_id))
                {
                    targetDictionary[cls_id].transform.position = newPosn;
                    namePlate = GameObject.Find($"{cls_id}_label");
                }
                else
                {
                    GameObject newObj = null;
                    if (target.label_name.Equals("vehicle"))
                    {
                        newObj = Instantiate(car, newPosn, Quaternion.identity);
                    }
                    else if (target.label_name.Equals("person"))
                    {
                        newObj = Instantiate(person, newPosn, Quaternion.identity);
                    } else
                    {
                        // add more objects here instead
                        newObj = Instantiate(car, newPosn, Quaternion.identity);
                    }
                    
                    newObj.SetActive(true);
                    newObj.name = cls_id;
                    targetDictionary[cls_id] = newObj;
                    namePlate = new GameObject($"{cls_id}_label");
                    namePlate.AddComponent<TextMesh>();
                }

                // draw label
                TextMesh textMesh = namePlate.GetComponent<TextMesh>();
                if (textMesh != null)
                {
                    textMesh.transform.position = targetDictionary[cls_id].transform.position + new Vector3(0, 1f, 0);
                    textMesh.transform.LookAt(Camera.main.transform);
                    textMesh.transform.rotation *= new Quaternion(0, 180f, 0, 0); ;
                    textMesh.characterSize = 0.05f;
                    textMesh.fontSize = 100;
                    textMesh.alignment = TextAlignment.Center;
                    textMesh.anchor = TextAnchor.MiddleCenter;
                    textMesh.color = Color.white;
                    textMesh.text = cls_id;
                }
                //namePlate.transform.parent = targetDictionary[target.cls_id].transform;
            }

            // TODO optimize
            List<string> toRemove = new List<string>();
            foreach (string key in targetDictionary.Keys)
            {
                if (key.Equals("connected_999")) continue; // TODO remove - debugging

                bool contains = false;
                foreach (JsonObject target in tracks)
                {
                    string cls_id = $"{target.label_name}_{target.track_id}";
                    if (key == cls_id)
                    {
                        contains = true;
                        break;
                    }
                }

                if (!contains)
                {
                    Destroy(targetDictionary[key]);
                    toRemove.Add(key);
                    GameObject namePlate = GameObject.Find($"{key}_label");
                    Destroy(namePlate);
                }
            }

            foreach (string key in toRemove)
            {
                targetDictionary.Remove(key);
            }
        }
    }

    /// <summary> 	
    /// Runs in background clientReceiveThread; Listens for incomming data. 	
    /// </summary>     
    private void ListenForData()
    {
        TcpListener server = new TcpListener(IPAddress.Any, 8080);
        server.Start();
        byte[] bytes = new byte[1024];
        while (true)
        {
            TcpClient client = null;
            try
            {
                print("Waiting for a connection...");
                client = server.AcceptTcpClient();
                print("Connected.");

                // Get a stream object for reading 				
                using (NetworkStream stream = client.GetStream())
                {
                    string hello = "Connected to Hololens.";
                    stream.Write(Encoding.ASCII.GetBytes(hello), 0, hello.Length);
                    List<JsonObject> jsonObjects = null;

                    // TODO remove - debugging
                    lock (json_obj.tracks)
                    {
                        JsonUtility.FromJsonOverwrite("{\"frame_num\": 210, \"tracks\": [{\"roi\": [537, 431, 590, 461], \"track_id\": 999, \"tracklet_id\": 999, \"label\": 1, \"label_name\": \"connected\", \"confidence\": 0.9053571564810616, \"top_attributes\": null, \"metadata\": null, \"point\": [0.0, 0.0, 80.0]}]}", json_obj);
                        jsonObjects = new List<JsonObject>(json_obj.tracks);
                        dispatcher.Invoke(() => UpdateScene(jsonObjects));
                    }

                    int length;
                    string serverMessage = "";

                    // reconstruct input stream				
                    while (stream.CanRead && (length = stream.Read(bytes, 0, bytes.Length)) != 0)
                    {
                        if (!stream.CanWrite) break;

                        // Convert byte array to string message.
                        string rcvd = Encoding.ASCII.GetString(bytes, 0, length);
                        print($"RCVD: {rcvd}");
                        serverMessage += rcvd;

                        // grabs the most recent actual json object and parses
                        string[] lines = serverMessage.Split("\n");
                        if (lines.Length < 2) { continue; }
                        string jsonMessage = lines[lines.Length - 2];
                        serverMessage = lines[lines.Length - 1];

                        try
                        {
                            lock (json_obj.tracks)
                            {
                                // add update scene task to dispatcher to execute on main thread
                                JsonUtility.FromJsonOverwrite(jsonMessage, json_obj);
                                jsonObjects = new List<JsonObject>(json_obj.tracks);
                                dispatcher.Invoke(() => UpdateScene(jsonObjects));
                            }
                        }
                        catch (ArgumentException e)
                        {
                            print(e);
                            continue;
                        }
                    }

                    if (stream.CanWrite)
                    {
                        string goodbye = "Done reading! Disconnecting.";
                        stream.Write(Encoding.ASCII.GetBytes(goodbye), 0, goodbye.Length);
                        client.Close();
                    }
                }
            }
            catch (SocketException socketException)
            {
                System.Diagnostics.Debug.WriteLine("SocketException: " + socketException);
                client.Close();
            }
            catch (IOException ioException)
            {
                System.Diagnostics.Debug.WriteLine("IOException: " + ioException);
            }
        }
        server.Stop();
    }
}
