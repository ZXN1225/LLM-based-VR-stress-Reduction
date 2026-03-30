using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.InputSystem;
using System.Collections;
using System.Collections.Generic;
using Newtonsoft.Json;

public class VRTherapyManager : MonoBehaviour
{
    [Header("Web Sync Settings")]
    public string pollUrl = "http://localhost:8000/get-latest-session";
    public float pollInterval = 3.0f;
    private string lastProcessedPlanJson = "";

    [Header("Required References")]
    public Material skyboxMaterial;
    public Light directionalLight;
    public AudioSource audioSource;
    public Camera mainCamera; 

    [Header("Therapy Settings")]
    public float sceneDuration = 60.0f; 
    public float transitionDuration = 3.0f;

    [Header("Keyboard Controls")]
    public KeyCode nextSceneKey = KeyCode.RightArrow; 
    public KeyCode prevSceneKey = KeyCode.LeftArrow;  
    public KeyCode nextMusicKey = KeyCode.M;          

    private List<SceneData> scenes = new List<SceneData>();
    private List<MusicData> playlist = new List<MusicData>();

    private int currentSceneIndex = 0;
    private bool skipRequested = false;
    private Coroutine therapyLoopCoroutine;
    private Coroutine musicLoopCoroutine;

    [System.Serializable]
    public class SceneData
    {
        public int step; public float duration; public string image_url; public UnityConfig unity_config;
    }
    [System.Serializable]
    public class UnityConfig
    {
        public float kelvin; public float intensity;
    }
    [System.Serializable]
    public class MusicData { public string audio_url; }

    void Start()
    {
        if (mainCamera != null)
        {
            mainCamera.transform.position = new Vector3(0, 1.2f, 0);
            mainCamera.transform.rotation = Quaternion.identity;
        }

        StartCoroutine(WebSyncLoop());
    }

    void Update()
    {
        if (Keyboard.current != null)
        {
            if (Keyboard.current.rightArrowKey.wasPressedThisFrame)
            {
                RequestNextScene(1);
            }

            if (Keyboard.current.leftArrowKey.wasPressedThisFrame)
            {
                RequestNextScene(-1);
            }

            if (Keyboard.current.mKey.wasPressedThisFrame)
            {
                SwitchMusic();
            }
        }
    }

    private void RequestNextScene(int direction)
    {
        skipRequested = true;
        currentSceneIndex = Mathf.Clamp(currentSceneIndex + direction, 0, scenes.Count - 1);
    }

    private void SwitchMusic()
    {
        if (musicLoopCoroutine != null) StopCoroutine(musicLoopCoroutine);
        musicLoopCoroutine = StartCoroutine(MusicPlayerLoop(true)); 
    }

    IEnumerator WebSyncLoop()
    {
        while (true)
        {
            using (UnityWebRequest request = UnityWebRequest.Get(pollUrl))
            {
                yield return request.SendWebRequest();
                if (request.result == UnityWebRequest.Result.Success)
                {
                    string jsonResponse = request.downloadHandler.text;
                    if (!string.IsNullOrEmpty(jsonResponse) && jsonResponse != "null")
                    {
                        var responseData = JsonConvert.DeserializeObject<Dictionary<string, object>>(jsonResponse);
                        string currentPlanJson = responseData["intervention_plan"].ToString();

                        if (currentPlanJson != lastProcessedPlanJson)
                        {
                            lastProcessedPlanJson = currentPlanJson;
                            scenes = JsonConvert.DeserializeObject<List<SceneData>>(currentPlanJson);
                            playlist = JsonConvert.DeserializeObject<List<MusicData>>(responseData["music_playlist"].ToString());

                            currentSceneIndex = 0;
                            if (therapyLoopCoroutine != null) StopCoroutine(therapyLoopCoroutine);
                            therapyLoopCoroutine = StartCoroutine(ExecuteTherapyLoop());

                            if (musicLoopCoroutine != null) StopCoroutine(musicLoopCoroutine);
                            musicLoopCoroutine = StartCoroutine(MusicPlayerLoop(false));
                        }
                    }
                }
            }
            yield return new WaitForSeconds(pollInterval);
        }
    }

    IEnumerator ExecuteTherapyLoop()
    {
        while (currentSceneIndex < scenes.Count)
        {
            var scene = scenes[currentSceneIndex];
            Debug.Log($"🎨 Loading" +
                $" {scene.step} / 10");

            using (UnityWebRequest texRequest = UnityWebRequestTexture.GetTexture(scene.image_url))
            {
                yield return texRequest.SendWebRequest();
                if (texRequest.result == UnityWebRequest.Result.Success)
                {
                    Texture2D newTex = DownloadHandlerTexture.GetContent(texRequest);
                    newTex.filterMode = FilterMode.Trilinear; 
                    newTex.anisoLevel = 8;                   
                    newTex.wrapMode = TextureWrapMode.Repeat;
                    
                    yield return StartCoroutine(TransitionRoutine(newTex, scene.unity_config));
 
                    float timer = 0;
                    skipRequested = false;
                    while (timer < sceneDuration && !skipRequested)
                    {
                        timer += Time.deltaTime;
                        yield return null;
                    }
                }
            }
 
            if (!skipRequested) currentSceneIndex++;
        }
    }

    IEnumerator TransitionRoutine(Texture2D newTex, UnityConfig config)
    {
        Texture prevTex = skyboxMaterial.GetTexture("_Tex2");
        if (prevTex != null) skyboxMaterial.SetTexture("_Tex1", prevTex);
        else skyboxMaterial.SetTexture("_Tex1", newTex);

        skyboxMaterial.SetTexture("_Tex2", newTex);
        skyboxMaterial.SetFloat("_Blend", 0);

        float elapsed = 0;
        float startKelvin = directionalLight.colorTemperature;
        float startIntensity = directionalLight.intensity;

        while (elapsed < transitionDuration)
        {
            elapsed += Time.deltaTime;
            float t = Mathf.SmoothStep(0, 1, elapsed / transitionDuration);
            skyboxMaterial.SetFloat("_Blend", t);
            if (directionalLight != null)
            {
                directionalLight.colorTemperature = Mathf.Lerp(startKelvin, config.kelvin, t);
                directionalLight.intensity = Mathf.Lerp(startIntensity, config.intensity, t);
            }
            yield return null;
        }

        skyboxMaterial.SetFloat("_Blend", 1.0f);
        if (prevTex != null && prevTex is Texture2D t2d) Destroy(t2d); 
    }

    IEnumerator MusicPlayerLoop(bool forceNext)
    {
        int musicIndex = forceNext ? 1 : 0;  
        while (true)
        {
            var music = playlist[musicIndex % playlist.Count];
            using (UnityWebRequest www = UnityWebRequestMultimedia.GetAudioClip(music.audio_url, AudioType.UNKNOWN))
            {
                yield return www.SendWebRequest();
                if (www.result == UnityWebRequest.Result.Success)
                {
                    AudioClip clip = DownloadHandlerAudioClip.GetContent(www);
                    audioSource.clip = clip;
                    audioSource.Play();
                    yield return new WaitForSeconds(clip.length);
                }
            }
            musicIndex++;
        }
    }
}