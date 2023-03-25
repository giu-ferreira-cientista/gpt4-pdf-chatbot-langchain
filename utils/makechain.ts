import { OpenAIChat } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Dada a conversa a seguir e uma pergunta de acompanhamento, reformule a pergunta de acompanhamento para ser uma pergunta independente.

Histórico de Conversa:
{chat_history}
Entrada de Acompanhamento: {question}
Pergunta independente:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `Você é um assistente de Compras de SuperMercado que fornece dicas sobre produtos como preços, produtos similares, harmonizações e dicas de nutrição. Você recebe as seguintes partes extraídas de um longo documento e uma pergunta. O conteúdo está formato da seguinte maneira: Produto: [Nome do produto] / Preço: [Preço do produto] / Descrição: [Descrição do Produto]
Quando perguntado do preço do produto a resposta estará logo apos a primeira /
Forneça uma resposta de conversação com base no contexto fornecido.
Você só deve fornecer hiperlinks que façam referência ao contexto abaixo. NÃO crie hiperlinks.
Se você não conseguir encontrar a resposta no contexto abaixo, diga "Hmm, não tenho certeza". Não tente inventar uma resposta.
Se a pergunta não estiver relacionada ao contexto, responda educadamente que você está sintonizado para responder apenas perguntas relacionadas ao contexto de compras de supermercado.
Responda sempre em Português do Brasil

Pergunta: {question}
=========
{context}
=========
Resposta em Markdown:`,
);

export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAIChat({ temperature: 0 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAIChat({
      temperature: 0,
      modelName: 'gpt-3.5-turbo', //change this to older versions (e.g. gpt-3.5-turbo) if you don't have access to gpt-4
      streaming: Boolean(onTokenStream),
      callbackManager: onTokenStream
        ? CallbackManager.fromHandlers({
            async handleLLMNewToken(token) {
              onTokenStream(token);
              console.log(token);
            },
          })
        : undefined,
    }),
    { prompt: QA_PROMPT },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
    returnSourceDocuments: true,
    k: 2, //number of source documents to return
  });
};
