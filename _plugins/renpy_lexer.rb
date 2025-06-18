
module Rouge
  module Lexers
    class Renpy < RegexLexer
      tag 'renpy'
      aliases 'rpy'
      filenames '*.rpy'
      mimetypes 'text/x-renpy'

      # Ren'Py 关键字
      keywords = %w(
        label menu jump call return if elif else while pass
        scene show hide with play stop queue
        init python early late define default
        transform animate at onlayer
        screen layer
      )

      functions = %w(
        say narrator play sound music stop
        scene show hide with transition
        pause
      )

      state :root do

        rule /#.*?$/, Comment::Single


        rule /""".*?"""/m, Str


        rule /"(?:\\"|[^"])*"/, Str
        rule /'(?:\\'|[^'])*'/, Str


        rule /\b\d+(\.\d+)?\b/, Num


        rule /\b(#{keywords.join('|')})\b/, Keyword


        rule /\b(#{functions.join('|')})\b/, Name::Builtin


        rule /\$[a-zA-Z_]\w*/, Name::Variable
        rule /[a-zA-Z_]\w*:/, Name::Label


        rule /(python)(\s*:)/ do
          groups Keyword, Punctuation
          push :python_code
        end


        rule /\\[\\'"nrt]/, Str::Escape


        rule /\s+/, Text
      end

      state :python_code do
        rule /\n/, Text, :pop!
        rule /.*?(?=\n)/m do
          delegate Python
        end
      end
    end
  end
end