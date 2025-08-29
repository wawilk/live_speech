# Create a voice live real-time voice agent (Preview)

This sample app uses voice live with generative AI and Azure AI Speech in the Azure AI Foundry portal.

You create and run an application to use voice live directly with generative AI models for real-time voice agents.

Using models directly allows specifying custom instructions (prompts) for each session, offering more flexibility for dynamic or experimental use cases.

Models may be preferable over agents when you want fine-grained control over session parameters or need to frequently adjust the prompt or configuration without updating an agent in the portal.

To use voice live, you don't need to deploy an audio model with your Azure AI Foundry resource. Voice live is fully managed, and the model is automatically deployed for you. For more information about models availability, see the voice live overview documentation.

### Microsoft Entra ID prerequisites

For the recommended keyless authentication with Microsoft Entra ID, you need to:

Install the Azure CLI used for keyless authentication with Microsoft Entra ID.
Assign the Cognitive Services User role to your user account. You can assign roles in the Azure portal under Access control (IAM) > Add role assignment.

### Set up

1. Create a new folder voice-live-quickstart and go to the quickstart folder with the following   command:

   ```
   mkdir voice-live-quickstart && cd voice-live-quickstart

   ```

2. Create a virtual environment. If you already have Python 3.10 or higher installed, you can create a virtual environment using the following commands:

   ```
   py -3 -m venv .venv
   .venv\scripts\activate
   ```

   Activating the Python environment means that when you run python or pip from the command line, you then use the Python interpreter contained in the .venv folder of your application. You can use the deactivate command to exit the python virtual environment, and can later reactivate it when needed.

   It is recommended that you create and activate a new Python environment to use to install the packages you need for this tutorial. Don't install packages into your global python installation. You should always use a virtual or conda environment when installing python packages, otherwise you can break your global installation of Python.
3. Install the packages:

   ```
   pip install -r requirements.txt

   ```

### Retrieve resource information

Rename the sample.env to .env

In the .env file, set the values for the following environment variables for authentication:

```
AZURE_VOICE_LIVE_ENDPOINT=your_endpointV  
OICE_LIVE_MODEL=your_model  
AZURE_VOICE_LIVE_API_VERSION=2025-05-01-preview
AZURE_VOICE_LIVE_API_KEY=your_api_key # Only required if using API key authentication

```

### Start a conversation

The sample code in this quickstart uses Microsoft Entra ID for the recommended keyless authentication.   If you prefer to use an API key, you can set the api_key variable instead of the token variable.


```
client = AzureVoiceLive(
    azure_endpoint = endpoint,
    api_version = api_version,
    token = token.token,
    # api_key = api_key,
)
```
