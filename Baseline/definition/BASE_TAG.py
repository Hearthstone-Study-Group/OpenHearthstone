class BASE_TAG:
    keyword_map = None
    
    def __getitem__(self, value):
        if self.keyword_map is None:
            self.keyword_map = {}
            for attr_name in dir(self):
                if not attr_name.startswith('__') and attr_name.upper() == attr_name:
                    self.keyword_map[getattr(self, attr_name)] = attr_name
        if value in self.keyword_map:
            return self.keyword_map[value]
        return None