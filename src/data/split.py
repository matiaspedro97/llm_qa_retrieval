from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocSplitter:
    def __init__(
            self,
            chunk_size: int = 100, 
            chunk_overlap: int = 20, 
            length_function: callable = len, 
            add_start_index: bool = True
    ) -> None:
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            length_function=length_function, 
            add_start_index=add_start_index
        )

    def split_documents(self, documents: list):
        # with page content
        texts = self.text_splitter.create_documents(documents)
        
        # string-only
        texts_ = [txt.page_content for txt in texts]
        return texts_